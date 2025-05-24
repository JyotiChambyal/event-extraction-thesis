# train.py

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import random
import os
from tqdm import tqdm

import src.config as config
from src.utils import generate_label_maps, WeightedCELoss # Add FocalLoss if using
from src.data_processor import load_and_process_all_data, EventExtractionDataset
from src.models import HistoryAwareTriggerModel, HistoryAwareArgumentModel
from src.evaluation import evaluate, calculate_f1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train():
    set_seed(config.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': config.ADDITIONAL_SPECIAL_TOKENS})

    # 2. Generate Label Maps
    data_dirs_for_labels = [config.TRAIN_DATA_DIR, config.DEV_DATA_DIR, config.TEST_DATA_DIR]
    trigger_label_map, argument_label_map, event_type_map, id_to_trigger_label, id_to_argument_label = \
        generate_label_maps(data_dirs_for_labels)

    # 3. Load and Process Data
    train_trigger_dataset, train_argument_dataset = load_and_process_all_data(
        [config.TRAIN_DATA_DIR], tokenizer, trigger_label_map, argument_label_map, event_type_map
    )
    dev_trigger_dataset, dev_argument_dataset = load_and_process_all_data(
        [config.DEV_DATA_DIR], tokenizer, trigger_label_map, argument_label_map, event_type_map
    )

    train_trigger_dataloader = DataLoader(train_trigger_dataset, sampler=RandomSampler(train_trigger_dataset), batch_size=config.BATCH_SIZE)
    train_argument_dataloader = DataLoader(train_argument_dataset, sampler=RandomSampler(train_argument_dataset), batch_size=config.BATCH_SIZE)

    dev_trigger_dataloader = DataLoader(dev_trigger_dataset, sampler=SequentialSampler(dev_trigger_dataset), batch_size=config.BATCH_SIZE)
    dev_argument_dataloader = DataLoader(dev_argument_dataset, sampler=SequentialSampler(dev_argument_dataset), batch_size=config.BATCH_SIZE)

    # Calculate Class Weights for Imbalance (for CrossEntropyLoss/FocalLoss)
    # This requires iterating through your dataset to count label frequencies.
    def calculate_class_weights(dataset, label_map):
        label_counts = defaultdict(int)
        for encoding in dataset.encodings:
            for label_id in encoding['labels']:
                if label_id != -100: # Ignore index
                    label_counts[label_id] += 1
        
        total_labels = sum(label_counts.values())
        if total_labels == 0: return None

        weights = torch.zeros(len(label_map))
        for label_id, count in label_counts.items():
            weights[label_id] = total_labels / (count + 1e-5) # Inverse frequency + smoothing
        # Normalize weights, optional but can help
        weights = weights / weights.sum() * len(label_map)
        return weights.to(device)

    trigger_class_weights = calculate_class_weights(train_trigger_dataset, trigger_label_map)
    argument_class_weights = calculate_class_weights(train_argument_dataset, argument_label_map)
    print(f"Trigger Class Weights: {trigger_class_weights}")
    print(f"Argument Class Weights: {argument_class_weights}")


    # 4. Initialize Models
    trigger_model = HistoryAwareTriggerModel(len(trigger_label_map), tokenizer).to(device)
    argument_model = HistoryAwareArgumentModel(len(argument_label_map), tokenizer).to(device)

    # 5. Define Optimizers and Schedulers
    # Separate parameters for BERT vs. classifier/CRF for different learning rates
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters_trigger = [
        {'params': [p for n, p in trigger_model.named_parameters() if not any(nd in n for nd in no_decay) and "crf" not in n and "classifier" not in n], 'weight_decay': config.WEIGHT_DECAY, 'lr': config.LEARNING_RATE},
        {'params': [p for n, p in trigger_model.named_parameters() if any(nd in n for nd in no_decay) and "crf" not in n and "classifier" not in n], 'weight_decay': 0.0, 'lr': config.LEARNING_RATE},
        {'params': [p for n, p in trigger_model.named_parameters() if "classifier" in n or "crf" in n], 'lr': config.LEARNING_RATE * config.CRF_LR_MULTIPLIER},
    ]
    optimizer_trigger = AdamW(optimizer_grouped_parameters_trigger, eps=config.ADAM_EPSILON)
    scheduler_trigger = get_linear_schedule_with_warmup(optimizer_trigger, num_warmup_steps=config.WARMUP_STEPS, num_training_steps=len(train_trigger_dataloader) * config.NUM_EPOCHS)

    optimizer_grouped_parameters_argument = [
        {'params': [p for n, p in argument_model.named_parameters() if not any(nd in n for nd in no_decay) and "crf" not in n and "classifier" not in n], 'weight_decay': config.WEIGHT_DECAY, 'lr': config.LEARNING_RATE},
        {'params': [p for n, p in argument_model.named_parameters() if any(nd in n for nd in no_decay) and "crf" not in n and "classifier" not in n], 'weight_decay': 0.0, 'lr': config.LEARNING_RATE},
        {'params': [p for n, p in argument_model.named_parameters() if "classifier" in n or "crf" in n], 'lr': config.LEARNING_RATE * config.CRF_LR_MULTIPLIER},
    ]
    optimizer_argument = AdamW(optimizer_grouped_parameters_argument, eps=config.ADAM_EPSILON)
    scheduler_argument = get_linear_schedule_with_warmup(optimizer_argument, num_warmup_steps=config.WARMUP_STEPS, num_training_steps=len(train_argument_dataloader) * config.NUM_EPOCHS)


    # 6. Training Loop
    best_dev_f1 = -1.0
    global_step = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")
        trigger_model.train()
        argument_model.train()
        total_trigger_loss = 0
        total_argument_loss = 0

        # Training Trigger Model
        print("Training Trigger Model...")
        for step, batch in enumerate(tqdm(train_trigger_dataloader, desc="Trigger Training")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            loss, _ = trigger_model(input_ids, attention_mask, token_type_ids, labels=labels, class_weights=trigger_class_weights)

            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            total_trigger_loss += loss.item()

            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trigger_model.parameters(), config.MAX_GRAD_NORM)
                optimizer_trigger.step()
                scheduler_trigger.step()
                optimizer_trigger.zero_grad()
                global_step += 1

        print(f"Avg Trigger Loss: {total_trigger_loss / len(train_trigger_dataloader):.4f}")

        # Training Argument Model
        print("Training Argument Model...")
        for step, batch in enumerate(tqdm(train_argument_dataloader, desc="Argument Training")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            loss, _ = argument_model(input_ids, attention_mask, token_type_ids, labels=labels, class_weights=argument_class_weights)

            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            total_argument_loss += loss.item()

            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(argument_model.parameters(), config.MAX_GRAD_NORM)
                optimizer_argument.step()
                scheduler_argument.step()
                optimizer_argument.zero_grad()
                global_step += 1

        print(f"Avg Argument Loss: {total_argument_loss / len(train_argument_dataloader):.4f}")

        # 7. Evaluation on Dev Set
        print("Evaluating on Dev Set...")
        dev_results = evaluate(
            trigger_model,
            argument_model,
            dev_trigger_dataloader,
            dev_argument_dataloader, # Note: this is simplified, actual eval needs conversation context
            tokenizer,
            id_to_trigger_label,
            id_to_argument_label,
            device
        )

        dev_event_f1 = dev_results["overall_event_f1"]

        # Save best model
        if dev_event_f1 > best_dev_f1:
            best_dev_f1 = dev_event_f1
            print(f"New best Dev Event F1: {best_dev_f1:.4f}. Saving models...")
            torch.save(trigger_model.state_dict(), os.path.join(config.OUTPUT_DIR, "best_trigger_model.pth"))
            torch.save(argument_model.state_dict(), os.path.join(config.OUTPUT_DIR, "best_argument_model.pth"))
        else:
            print(f"Dev Event F1 did not improve. Best so far: {best_dev_f1:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    train()