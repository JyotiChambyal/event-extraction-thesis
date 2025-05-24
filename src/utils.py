# src/utils.py

import os
import json
from typing import List, Dict, Tuple, Set, Optional
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

def generate_label_maps(data_dirs: List[str]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str], Dict[int, str], Dict[int, str]]:
    """
    Scans the dataset to identify all unique trigger, argument, and event types
    and generates corresponding label-to-ID mappings.
    Returns:
        A tuple of (trigger_label_map, argument_label_map, event_type_map,
                    id_to_trigger_label, id_to_argument_label)
    """
    unique_trigger_labels: Set[str] = set()
    unique_argument_labels: Set[str] = set()
    unique_event_types: Set[str] = set()

    for data_dir in data_dirs:
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)

                for turn_key, turn_data in conversation.get("events", {}).items():
                    for event_type_full, event_details in turn_data.items():
                        base_event_type = event_type_full.split(':')[0]
                        unique_event_types.add(base_event_type)

                        if 'labels' in event_details and event_details['labels']:
                            for label_sequence in event_details['labels']:
                                for full_label_str in label_sequence:
                                    clean_label = full_label_str.split(': ')[-1] if ': ' in full_label_str else full_label_str

                                    if clean_label == "O":
                                        unique_trigger_labels.add(clean_label)
                                        unique_argument_labels.add(clean_label)
                                    elif clean_label.startswith(("B-", "I-", "S-")):
                                        tag_prefix = clean_label.split('-')[0]
                                        name = clean_label.split('-', 1)[1]

                                        if name in unique_event_types: # Heuristic: if name matches event type, it's a trigger
                                            unique_trigger_labels.add(f"{tag_prefix}-{name}")
                                        else: # Otherwise, assume it's an argument role
                                            unique_argument_labels.add(f"{tag_prefix}-{name}")
                                    else:
                                        print(f"Warning: Unrecognized label format '{full_label_str}' in {filepath}")

    # Generate mappings
    trigger_label_map = {"O": 0}
    current_id = 1
    for label in sorted(list(unique_trigger_labels)):
        if label != "O":
            trigger_label_map[label] = current_id
            current_id += 1
    id_to_trigger_label = {v: k for k, v in trigger_label_map.items()}

    argument_label_map = {"O": 0}
    current_id = 1
    for label in sorted(list(unique_argument_labels)):
        if label != "O":
            argument_label_map[label] = current_id
            current_id += 1
    id_to_argument_label = {v: k for k, v in argument_label_map.items()}

    event_type_map = {event_type: event_type.lower().replace("_", "") for event_type in sorted(list(unique_event_types))} # Ensure placeholder is clean

    print(f"Generated Trigger Labels: {trigger_label_map}")
    print(f"Generated Argument Labels: {argument_label_map}")
    print(f"Generated Event Types: {event_type_map}")

    return trigger_label_map, argument_label_map, event_type_map, id_to_trigger_label, id_to_argument_label


def map_word_labels_to_tokens(
    tokenized_output: Dict, # Output from tokenizer(), includes input_ids, offset_mapping etc.
    original_sentence_words: List[str], # The list of words from the original sentence
    word_labels: List[str], # Word-level labels (BIO/BIS) corresponding to original_sentence_words
    label_map: Dict[str, int],
    ignore_index: int = -100,
    is_trigger_labels: bool = False # Flag to handle B/I/S logic
) -> List[int]:
    """
    Maps word-level labels to token-level labels, handling subword tokenization.
    Assigns ignore_index to special tokens and subwords after the first.
    """
    token_labels = []
    word_ids = tokenized_output.word_ids() # This links BERT tokens to original words (by index)

    previous_word_idx = None
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None: # Special tokens (CLS, SEP, etc.)
            token_labels.append(ignore_index)
        elif word_idx != previous_word_idx: # Start of a new word
            # Ensure the word_idx is within the bounds of original_sentence_words
            if word_idx < len(word_labels):
                original_label = word_labels[word_idx]
                if is_trigger_labels: # Only for triggers, ensure B-tag starts a span
                    if original_label.startswith("I-"): # Should ideally not happen in correct dataset
                        token_labels.append(label_map.get("B-" + original_label[2:], ignore_index))
                    else:
                        token_labels.append(label_map.get(original_label, ignore_index))
                else: # For arguments, standard BIO
                     token_labels.append(label_map.get(original_label, ignore_index))
            else: # Should not happen if word_ids are correctly generated for the target part
                token_labels.append(ignore_index)
        else: # Continuation of a word (subword token)
            if word_idx < len(word_labels):
                original_label = word_labels[word_idx]
                token_labels.append(label_map.get(original_label, ignore_index))
            else:
                token_labels.append(ignore_index)
        previous_word_idx = word_idx
    return token_labels


class WeightedCELoss(CrossEntropyLoss):
    """
    Custom CrossEntropyLoss that can apply class weights.
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, num_labels, sequence_length) -> Permute to (batch_size * sequence_length, num_labels)
        # Target: (batch_size, sequence_length) -> Flatten to (batch_size * sequence_length)
        input = input.permute(0, 2, 1).reshape(-1, input.size(1))
        target = target.reshape(-1)
        return super().forward(input, target)

# Focal Loss (Optional, can be added here if FOCAL_LOSS_ENABLED is True)
# Implementation of Focal Loss can be found online or custom written.
# Example signature: def FocalLoss(nn.Module): ...