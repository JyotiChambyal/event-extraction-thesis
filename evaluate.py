# evaluate.py

import torch
from torch.utils.data import DataLoader, SequentialSampler
import os

import src.config as config
from src.utils import generate_label_maps
from src.data_processor import load_and_process_all_data
from src.models import HistoryAwareTriggerModel, HistoryAwareArgumentModel
from src.evaluation import evaluate # Re-import evaluate function

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': config.ADDITIONAL_SPECIAL_TOKENS})

    # 2. Generate Label Maps (using all data to ensure completeness)
    data_dirs_for_labels = [config.TRAIN_DATA_DIR, config.DEV_DATA_DIR, config.TEST_DATA_DIR]
    trigger_label_map, argument_label_map, event_type_map, id_to_trigger_label, id_to_argument_label = \
        generate_label_maps(data_dirs_for_labels)

    # 3. Load Test Data
    test_trigger_dataset, test_argument_dataset = load_and_process_all_data(
        [config.TEST_DATA_DIR], tokenizer, trigger_label_map, argument_label_map, event_type_map
    )

    test_trigger_dataloader = DataLoader(test_trigger_dataset, sampler=SequentialSampler(test_trigger_dataset), batch_size=config.BATCH_SIZE)
    test_argument_dataloader = DataLoader(test_argument_dataset, sampler=SequentialSampler(test_argument_dataset), batch_size=config.BATCH_SIZE)


    # 4. Initialize Models and Load Checkpoints
    trigger_model = HistoryAwareTriggerModel(len(trigger_label_map), tokenizer).to(device)
    argument_model = HistoryAwareArgumentModel(len(argument_label_map), tokenizer).to(device)

    trigger_model_path = os.path.join(config.OUTPUT_DIR, "best_trigger_model.pth")
    argument_model_path = os.path.join(config.OUTPUT_DIR, "best_argument_model.pth")

    if not os.path.exists(trigger_model_path) or not os.path.exists(argument_model_path):
        print("Error: Trained models not found. Please run train.py first.")
        return

    trigger_model.load_state_dict(torch.load(trigger_model_path, map_location=device))
    argument_model.load_state_dict(torch.load(argument_model_path, map_location=device))

    print("Loaded trained models for evaluation.")

    # 5. Evaluate on Test Set
    test_results = evaluate(
        trigger_model,
        argument_model,
        test_trigger_dataloader,
        test_argument_dataloader, # Still simplified in `evaluate`
        tokenizer,
        id_to_trigger_label,
        id_to_argument_label,
        device,
        is_test_set=True
    )

    # You can save test_results to a file here
    print("\nTest Evaluation Complete.")

if __name__ == "__main__":
    main()