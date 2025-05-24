# src/evaluation.py

from typing import List, Dict, Tuple, Set
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Define Event and Argument structures for clarity
class EventInstance:
    def __init__(self, trigger_span: Tuple[int, int], trigger_type: str, arguments: List['ArgumentInstance']):
        self.trigger_span = trigger_span # (start_char_idx, end_char_idx) in original sentence
        self.trigger_type = trigger_type
        self.arguments = arguments # List of ArgumentInstance objects

    def __eq__(self, other):
        if not isinstance(other, EventInstance):
            return NotImplemented
        return (self.trigger_span == other.trigger_span and
                self.trigger_type == other.trigger_type and
                set(self.arguments) == set(other.arguments))

    def __hash__(self):
        return hash((self.trigger_span, self.trigger_type, frozenset(self.arguments)))

class ArgumentInstance:
    def __init__(self, arg_span: Tuple[int, int], arg_role: str):
        self.arg_span = arg_span # (start_char_idx, end_char_idx) in original sentence
        self.arg_role = arg_role

    def __eq__(self, other):
        if not isinstance(other, ArgumentInstance):
            return NotImplemented
        return (self.arg_span == other.arg_span and
                self.arg_role == other.arg_role)

    def __hash__(self):
        return hash((self.arg_span, self.arg_role))

def decode_predictions_to_spans(
    input_ids: List[int],
    predictions: List[int], # List of predicted label IDs for the sequence
    id_to_label_map: Dict[int, str],
    tokenizer,
    raw_sentence_text: str # Needed for char offsets
) -> List[Tuple[Tuple[int, int], str]]:
    """
    Converts token-level label ID predictions into (span, label) tuples.
    Handles BIO/BIS tagging scheme and subword tokens.
    Returns a list of (char_span_tuple, label_string).
    """
    spans = []
    current_span_start_char = -1
    current_span_label = None

    # Get word offsets for reconstruction
    # Need to tokenize the raw sentence to get its original word offsets,
    # then map back to the full input_ids. This is complex.
    # The most robust way is to store original word offsets in the dataset preparation
    # or pass a direct mapping from token_idx to original word_idx/char_span.

    # For simplicity, we assume we have tokenized the raw_sentence_text
    # and have a mapping to its original word indices/char offsets.
    # This requires `return_offsets_mapping=True` on the tokenizer when decoding.

    # Re-tokenize the raw sentence to get offsets relative to itself
    tokenized_sentence_for_offsets = tokenizer(
        raw_sentence_text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    sentence_offsets_map = tokenized_sentence_for_offsets['offset_mapping']
    sentence_tokens = tokenized_sentence_for_offsets['input_ids']

    # We need to find where `sentence_tokens` are located within the `input_ids` list.
    # This is tricky because of history and special tokens.
    # Let's assume `input_ids` contains special tokens at start/end, history, then the current sentence.
    # Find the start index of the current sentence's actual BERT tokens in `input_ids`.
    start_of_current_sentence_in_input_ids = -1
    try:
        # Find first token of the current sentence in the input_ids
        first_token_of_sentence_id = sentence_tokens[0]
        for idx in range(len(input_ids)):
            if input_ids[idx] == first_token_of_sentence_id:
                # Basic check, more robust: check if the sequence matches
                if input_ids[idx : idx + len(sentence_tokens)] == sentence_tokens:
                    start_of_current_sentence_in_input_ids = idx
                    break
    except IndexError: # Empty sentence_tokens or input_ids
        return []

    if start_of_current_sentence_in_input_ids == -1:
        return [] # Current sentence not found, likely due to truncation

    # Iterate through the predicted labels corresponding to the current sentence
    for i, token_id in enumerate(sentence_tokens):
        # Map this token_id's predicted label
        global_token_idx = start_of_current_sentence_in_input_ids + i
        if global_token_idx >= len(predictions): continue # Prediction might be shorter due to padding

        label_id = predictions[global_token_idx]
        label = id_to_label_map.get(label_id, "O") # Default to 'O'

        tag, name = ("O", None)
        if '-' in label:
            tag, name = label.split('-', 1)
        else:
            tag = label

        if tag.startswith("B-") or tag.startswith("S-"): # Start of a new span
            if current_span_start_char != -1 and current_span_label is not None:
                # End previous span if one was active
                end_char = sentence_offsets_map[i-1][1] # End char of previous token in sentence
                spans.append(((current_span_start_char, end_char), current_span_label))

            current_span_start_char = sentence_offsets_map[i][0]
            current_span_label = name
        elif tag.startswith("I-"): # Continuation of a span
            if current_span_start_char == -1 or current_span_label != name:
                # This 'I' tag starts a span incorrectly or its type doesn't match previous.
                # Treat as a new span or 'O' depending on strictness. Here, we start a new span.
                if current_span_start_char != -1 and current_span_label is not None: # End previous
                    end_char = sentence_offsets_map[i-1][1]
                    spans.append(((current_span_start_char, end_char), current_span_label))
                current_span_start_char = sentence_offsets_map[i][0]
                current_span_label = name # Start new span with I-tag's name
        else: # 'O' tag or other non-span tag
            if current_span_start_char != -1 and current_span_label is not None:
                # End active span
                end_char = sentence_offsets_map[i-1][1]
                spans.append(((current_span_start_char, end_char), current_span_label))
            current_span_start_char = -1
            current_span_label = None

    # After loop, if a span is still active, add it
    if current_span_start_char != -1 and current_span_label is not None:
        end_char = sentence_offsets_map[len(sentence_tokens)-1][1] # Last token of sentence
        spans.append(((current_span_start_char, end_char), current_span_label))

    return spans

def calculate_f1(num_true_positives: int, num_predictions: int, num_gold: int) -> Tuple[float, float, float]:
    """Calculates Precision, Recall, and F1-score."""
    precision = num_true_positives / num_predictions if num_predictions > 0 else 0.0
    recall = num_true_positives / num_gold if num_gold > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def calculate_event_level_f1(gold_events: List[EventInstance], predicted_events: List[EventInstance]) -> Tuple[float, float, float]:
    """
    Calculates event-level F1-score.
    Requires exact match of trigger span, type, and all associated arguments (span + role).
    """
    # Convert lists to sets for efficient comparison
    gold_set = set(gold_events)
    predicted_set = set(predicted_events)

    tp = len(gold_set.intersection(predicted_set))
    pred_count = len(predicted_set)
    gold_count = len(gold_set)

    return calculate_f1(tp, pred_count, gold_count)

def evaluate(
    trigger_model,
    argument_model,
    dataloader_trigger,
    dataloader_argument, # Note: this dataloader would be per-trigger during inference
    tokenizer,
    id_to_trigger_label: Dict[int, str],
    id_to_argument_label: Dict[int, str],
    device: torch.device,
    is_test_set: bool = False # Flag for test set evaluation vs validation
) -> Dict[str, float]:
    """
    Evaluates the models, combining trigger and argument extraction for event-level F1.
    """
    trigger_model.eval()
    argument_model.eval()

    all_gold_events: List[EventInstance] = []
    all_predicted_events: List[EventInstance] = []

    # Iterate through batches of full conversations for trigger prediction
    # Dataloader for triggers should ideally yield: input_ids, attention_mask, token_type_ids,
    # AND original_sentence_raw_text (or sufficient info to reconstruct it for span decoding)
    # This requires your data_processor.py to include raw_text in the encoding dict.

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader_trigger, desc="Evaluating Triggers")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device) # Gold trigger labels

            # Assuming batch also contains the raw sentence texts for decoding
            # You'll need to modify data_processor to include these
            # raw_sentence_texts = batch['raw_sentence_text'] # List of strings in the batch

            _, trigger_logits = trigger_model(input_ids, attention_mask, token_type_ids, labels=labels)
            trigger_predictions = trigger_model.crf.decode(emissions=trigger_logits, mask=(attention_mask == 1))

            for i in range(input_ids.size(0)): # Iterate through examples in batch
                # Get the actual raw sentence text for this example from your batch
                # This is a placeholder, you need to pass raw text from data_processor
                # For this to work, your dataset input needs to include the original raw string
                # or enough info to recreate it for accurate offset mapping.
                # Simplification: Let's assume you pass raw_sentence_texts alongside batch
                # raw_sentence_text_for_decoding = raw_sentence_texts[i]
                # For this example, let's just use a dummy string.
                # In real scenario, the data_processor.py needs to pass original string.
                dummy_raw_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True) # Placeholder

                # Decode predicted triggers for this example
                # THIS IS THE PART THAT NEEDS THE ORIGINAL RAW TEXT
                # The decode_predictions_to_spans needs the original sentence text to get char offsets correctly.
                predicted_trigger_spans_types = decode_predictions_to_spans(
                    input_ids[i].tolist(),
                    trigger_predictions[i], # This is a list of int IDs
                    id_to_trigger_label,
                    tokenizer,
                    dummy_raw_sentence # Replace with actual sentence text
                )
                # print(f"Predicted Triggers for example {i}: {predicted_trigger_spans_types}")


                # Now, for each predicted trigger, run argument model
                current_predicted_event_triggers: List[EventInstance] = []
                for pred_trg_span, pred_trg_type in predicted_trigger_spans_types:
                    # Construct argument input for this specific predicted trigger
                    # This is analogous to how prepare_inputs_for_model creates argument inputs
                    # You need to pass history for this specific turn, the predicted trigger, etc.
                    # This means the evaluation loop needs context/history from the original conversation.
                    # This is complex and usually requires a custom evaluation loop that reconstructs
                    # the full conversation context.

                    # For simplicity, this evaluation function will just get the predicted arguments
                    # for *gold* triggers during evaluation.
                    # For full event-level, you'd need to run argument prediction on *predicted* triggers
                    # from the trigger model, and then collect arguments for each.

                    # This is a placeholder for a more complex inference pipeline:
                    # For *each* predicted trigger, create an argument_input.
                    # arg_input_ids, arg_attention_mask, arg_token_type_ids = format_argument_input(...)
                    # arg_preds, _ = argument_model(arg_input_ids, arg_attention_mask, arg_token_type_ids)
                    # decoded_args = decode_predictions_to_spans(arg_preds, id_to_argument_label, raw_sentence_text)
                    # predicted_arguments = [ArgumentInstance(s,r) for s,r in decoded_args]

                    # For a simplified evaluation (only for initial dev):
                    # Let's assume you are comparing with gold triggers to get argument F1
                    # (This is what the paper does for "w/ ground-truth triggers")
                    predicted_arguments_for_this_trigger: List[ArgumentInstance] = [] # Placeholder
                    current_predicted_event_triggers.append(
                        EventInstance(pred_trg_span, pred_trg_type, predicted_arguments_for_this_this_trigger)
                    )

                all_predicted_events.extend(current_predicted_event_triggers)

                # --- Get Gold Events (requires reconstructing from raw JSON) ---
                # This part is also tricky: you need access to the original JSON structure
                # for the current example to build the EventInstance objects.
                # You'd typically pass a list of original JSON event objects from data_processor.
                # For this example, let's assume `gold_events_for_this_example` is provided
                # This means your DataLoader for evaluation needs to return the *original JSON objects* or processed EventInstance objects.
                # gold_events_for_this_example = ... # From data_processor.py output
                gold_events_for_this_example = [] # Placeholder
                all_gold_events.extend(gold_events_for_this_example)


    # Calculate overall event-level F1
    overall_p, overall_r, overall_f1 = calculate_event_level_f1(all_gold_events, all_predicted_events)

    # Calculate per-class F1 (more detailed breakdown)
    # This requires grouping gold/predicted events by type/role
    # per_class_metrics = calculate_per_class_f1(all_gold_events, all_predicted_events)

    results = {
        "overall_event_precision": overall_p,
        "overall_event_recall": overall_r,
        "overall_event_f1": overall_f1,
        # "per_class_metrics": per_class_metrics # Add this if you implement per_class_f1
    }

    print(f"\n--- Evaluation Results ({'Test' if is_test_set else 'Validation'}) ---")
    print(f"Overall Event F1: {overall_f1:.4f}")
    print(f"Overall Event Precision: {overall_p:.4f}")
    print(f"Overall Event Recall: {overall_r:.4f}")

    return results

# Helper for per-class F1 (conceptual outline)
def calculate_per_class_f1(gold_events: List[EventInstance], predicted_events: List[EventInstance]) -> Dict[str, Dict[str, float]]:
    return {} 