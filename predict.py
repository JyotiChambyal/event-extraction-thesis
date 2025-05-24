# predict.py

import torch
from transformers import AutoTokenizer
import os
from typing import List, Dict, Tuple

import src.config as config
from src.utils import generate_label_maps
from src.models import HistoryAwareTriggerModel, HistoryAwareArgumentModel
from src.evaluation import decode_predictions_to_spans, EventInstance, ArgumentInstance

def predict_event_instances(
    text_input: str,
    history_texts: List[str], # List of previous turn texts
    trigger_model,
    argument_model,
    tokenizer,
    id_to_trigger_label: Dict[int, str],
    id_to_argument_label: Dict[int, str],
    event_type_map: Dict[str, str],
    device: torch.device
) -> List[EventInstance]:
    """
    Predicts event instances (triggers and their arguments) from input text and history.
    """
    trigger_model.eval()
    argument_model.eval()

    predicted_events: List[EventInstance] = []

    with torch.no_grad():
        # --- 1. Predict Triggers ---
        # Format input for trigger model
        full_text_for_trigger = tokenizer.cls_token
        for h_text in history_texts:
            full_text_for_trigger += h_text + tokenizer.sep_token
        full_text_for_trigger += text_input + tokenizer.sep_token

        trigger_token_type_ids = [0] * (len(tokenizer.tokenize(full_text_for_trigger)) + 2) # Rough estimate
        current_sentence_start_char_in_full_text = full_text_for_trigger.find(text_input)

        trigger_encoding = tokenizer(
            full_text_for_trigger,
            is_split_into_words=False,
            padding='max_length',
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt",
            return_offsets_mapping=True # For accurate decoding
        )
        # Manually adjust token_type_ids
        for idx, (start_offset, end_offset) in enumerate(trigger_encoding['offset_mapping'][0]):
            if start_offset >= current_sentence_start_char_in_full_text and \
               end_offset <= current_sentence_start_char_in_full_text + len(text_input):
                if idx < len(trigger_token_type_ids):
                    trigger_token_type_ids[idx] = 1 # Mark as segment 1

        trigger_input_ids = trigger_encoding['input_ids'].to(device)
        trigger_attention_mask = trigger_encoding['attention_mask'].to(device)
        trigger_token_type_ids_tensor = torch.tensor(trigger_token_type_ids[:trigger_input_ids.size(1)], dtype=torch.long).unsqueeze(0).to(device)

        _, trigger_logits = trigger_model(trigger_input_ids, trigger_attention_mask, trigger_token_type_ids_tensor)
        trigger_predictions = trigger_model.crf.decode(emissions=trigger_logits, mask=(trigger_attention_mask == 1))

        # Decode predicted triggers
        predicted_trigger_spans_types = decode_predictions_to_spans(
            trigger_input_ids[0].tolist(),
            trigger_predictions[0],
            id_to_trigger_label,
            tokenizer,
            text_input # Pass the current sentence for decoding offsets
        )

        # --- 2. Predict Arguments for Each Predicted Trigger ---
        for pred_trg_span_char, pred_trg_type_str in predicted_trigger_spans_types:
            # Map char span to word indices if needed, or directly use char span
            # For this example, we'll directly use char_span.
            # Convert trigger type string to placeholder (e.g., "request_action")
            event_type_placeholder = event_type_map.get(pred_trg_type_str, "unknown_type")

            # Build argument model input for this trigger
            arg_full_text_parts = [tokenizer.cls_token, event_type_placeholder, "[TYPE]"]
            for h_text in history_texts:
                arg_full_text_parts.append(h_text)
                arg_full_text_parts.append(tokenizer.sep_token)

            # Insert [TRG] markers around the predicted trigger in the current sentence
            current_sentence_with_trg_markers = []
            char_idx = 0
            for word_token in tokenizer.tokenize(text_input): # Iterate original words of current sentence
                word = tokenizer.decode(tokenizer.encode(word_token, add_special_tokens=False))
                if char_idx == pred_trg_span_char[0]:
                    current_sentence_with_trg_markers.append("[TRG]")
                current_sentence_with_trg_markers.append(word)
                char_idx += len(word) + 1 # +1 for space
            if char_idx -1 == pred_trg_span_char[1]: # End of sentence, ensure last marker if needed
                 current_sentence_with_trg_markers.append("[/TRG]")

            arg_full_text_parts.extend(current_sentence_with_trg_markers)
            arg_full_text = " ".join(arg_full_text_parts)

            arg_encoding = tokenizer(
                arg_full_text,
                is_split_into_words=False,
                padding='max_length',
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt",
                return_offsets_mapping=True
            )

            # Adjust token_type_ids for argument model
            arg_token_type_ids = [0] * (len(tokenizer.tokenize(arg_full_text)) + 2)
            current_sentence_start_char_in_full_arg_text = arg_full_text.find(" ".join(current_sentence_with_trg_markers))
            for idx, (start_offset, end_offset) in enumerate(arg_encoding['offset_mapping'][0]):
                if start_offset >= current_sentence_start_char_in_full_arg_text and \
                   end_offset <= current_sentence_start_char_in_full_arg_text + len(" ".join(current_sentence_with_trg_markers)):
                    if idx < len(arg_token_type_ids):
                        arg_token_type_ids[idx] = 1

            arg_input_ids = arg_encoding['input_ids'].to(device)
            arg_attention_mask = arg_encoding['attention_mask'].to(device)
            arg_token_type_ids_tensor = torch.tensor(arg_token_type_ids[:arg_input_ids.size(1)], dtype=torch.long).unsqueeze(0).to(device)

            _, argument_logits = argument_model(arg_input_ids, arg_attention_mask, arg_token_type_ids_tensor)
            argument_predictions = argument_model.crf.decode(emissions=argument_logits, mask=(arg_attention_mask == 1))

            # Decode predicted arguments
            predicted_arg_spans_roles = decode_predictions_to_spans(
                arg_input_ids[0].tolist(),
                argument_predictions[0],
                id_to_argument_label,
                tokenizer,
                text_input # Again, pass the current sentence for decoding offsets
            )
            # Filter out arguments that fall within the trigger span itself
            filtered_arg_spans_roles = [
                (span, role) for span, role in predicted_arg_spans_roles
                if not (span[0] >= pred_trg_span_char[0] and span[1] <= pred_trg_span_char[1]) # Check for overlap
            ]

            predicted_arguments = [ArgumentInstance(s, r) for s, r in filtered_arg_spans_roles]
            predicted_events.append(EventInstance(pred_trg_span_char, pred_trg_type_str, predicted_arguments))

    return predicted_events

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': config.ADDITIONAL_SPECIAL_TOKENS})

    # 2. Generate Label Maps (from full dataset to ensure completeness)
    data_dirs_for_labels = [config.TRAIN_DATA_DIR, config.DEV_DATA_DIR, config.TEST_DATA_DIR]
    trigger_label_map, argument_label_map, event_type_map, id_to_trigger_label, id_to_argument_label = \
        generate_label_maps(data_dirs_for_labels)

    # 3. Initialize Models and Load Checkpoints
    trigger_model = HistoryAwareTriggerModel(len(trigger_label_map), tokenizer).to(device)
    argument_model = HistoryAwareArgumentModel(len(argument_label_map), tokenizer).to(device)

    trigger_model_path = os.path.join(config.OUTPUT_DIR, "best_trigger_model.pth")
    argument_model_path = os.path.join(config.OUTPUT_DIR, "best_argument_model.pth")

    if not os.path.exists(trigger_model_path) or not os.path.exists(argument_model_path):
        print("Error: Trained models not found. Please run train.py first.")
        return

    trigger_model.load_state_dict(torch.load(trigger_model_path, map_location=device))
    argument_model.load_state_dict(torch.load(argument_model_path, map_location=device))

    print("Loaded trained models for prediction.")

    # 4. Example Prediction Usage
    print("\n--- Example Prediction ---")
    current_email_text = "Can we please schedule a meeting about the project status next week?"
    # Simulate history from previous emails/turns
    email_history = [
        "Hi Team, This is a follow-up to our last discussion on the new project proposal.",
        "I need an update on the progress of task ID 12345."
    ]

    print(f"\nCurrent Email: {current_email_text}")
    print(f"History: {history_texts}")

    predicted_events = predict_event_instances(
        current_email_text,
        email_history,
        trigger_model,
        argument_model,
        tokenizer,
        id_to_trigger_label,
        id_to_argument_label,
        event_type_map,
        device
    )

    print("\nPredicted Events:")
    if predicted_events:
        for event in predicted_events:
            print(f"  Trigger: '{current_email_text[event.trigger_span[0]:event.trigger_span[1]]}' "
                  f"(Type: {event.trigger_type}, Span: {event.trigger_span})")
            if event.arguments:
                for arg in event.arguments:
                    print(f"    Argument: '{current_email_text[arg.arg_span[0]:arg.arg_span[1]]}' "
                          f"(Role: {arg.arg_role}, Span: {arg.arg_span})")
            else:
                print("    No arguments predicted for this event.")
    else:
        print("No events predicted.")

if __name__ == "__main__":
    main()