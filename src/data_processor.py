# src/data_processor.py

import os
import json
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast # Use PreTrainedTokenizerFast for offset_mapping robustness

from . import config 
from . import map_word_labels_to_tokens

class EventExtractionDataset(Dataset):
    """
    A custom PyTorch Dataset for event extraction tasks.
    """
    def __init__(self, encodings: List[Dict]):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.encodings[idx]
        return {key: torch.tensor(val) for key, val in encoding.items()}


def prepare_inputs_for_model(
    filepath: str,
    tokenizer: PreTrainedTokenizerFast,
    trigger_label_map: Dict[str, int],
    argument_label_map: Dict[str, int],
    event_type_map: Dict[str, str],
    max_length: int = config.MAX_SEQ_LENGTH
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepares input features for both Trigger and Argument Extraction Models
    from a single conversation JSON file, handling history and special tokens.
    """
    trigger_model_inputs = []
    argument_model_inputs = []

    with open(filepath, 'r', encoding='utf-8') as f:
        conversation = json.load(f)

    sentences = conversation["sentences"]
    events_data = conversation["events"]

    # History will be a list of raw texts of previous turns/emails in the thread
    history_raw_texts: List[str] = []

    for i, sentence_tokens in enumerate(sentences):
        turn_key = f"turn_{i}"
        current_sentence_raw = " ".join(sentence_tokens)

        # --- A. Prepare Input for Trigger Model ---
        # Input format: [CLS] H_1 [SEP] H_2 [SEP] ... [SEP] Xt [SEP]
        # Labels: BIS-O for current sentence (Xt)

        # Build the raw text string for tokenizer
        # This approach ensures `word_ids()` and `offset_mapping` are for the full string
        full_text_for_trigger = tokenizer.cls_token
        history_segment_ids = []
        for h_text in history_raw_texts:
            full_text_for_trigger += h_text + tokenizer.sep_token
            history_segment_ids.extend([0] * (len(tokenizer.tokenize(h_text)) + 1)) # +1 for SEP

        full_text_for_trigger += current_sentence_raw + tokenizer.sep_token
        current_segment_ids = [1] * len(tokenizer.tokenize(current_sentence_raw)) # For current sentence
        final_sep_segment_id = [1] # For final SEP token

        tokenized_output = tokenizer(
            full_text_for_trigger,
            is_split_into_words=False, # We've already joined the text
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_token_type_ids=True # Get segment IDs from tokenizer if it handles it
        )

        # Manually create token_type_ids to accurately reflect history (0) vs current (1)
        # This requires finding the start of the current sentence within the tokenized input.
        # A robust way is to mark the start of current_sentence_raw in the `full_text_for_trigger` string.
        current_sentence_start_char_in_full_text = full_text_for_trigger.find(current_sentence_raw)

        token_type_ids = [0] * len(tokenized_output['input_ids']) # Default to segment 0 (history)
        if current_sentence_start_char_in_full_text != -1:
            for token_idx, (start_offset, end_offset) in enumerate(tokenized_output['offset_mapping']):
                # Tokens that are part of the current sentence's raw text
                if start_offset >= current_sentence_start_char_in_full_text and \
                   end_offset <= current_sentence_start_char_in_full_text + len(current_sentence_raw):
                    token_type_ids[token_idx] = 1 # Mark as segment 1

        # Get gold trigger labels for the current sentence words
        current_sentence_gold_trigger_labels: List[str] = ["O"] * len(sentence_tokens)
        if turn_key in events_data:
            for event_type_full, event_details in events_data[turn_key].items():
                if 'labels' in event_details and event_details['labels']:
                    json_label_sequence = event_details['labels'][0] # Assuming first list for current turn
                    for word_idx, full_label_str in enumerate(json_label_sequence):
                        clean_label = full_label_str.split(': ')[-1] if ': ' in full_label_str else full_label_str
                        # Heuristic: if label implies event type (e.g., 'B-Request_Action'), it's a trigger label
                        base_name_from_label = clean_label.split('-',1)[1] if '-' in clean_label else ""
                        if clean_label == "O" or base_name_from_label in event_type_map:
                             current_sentence_gold_trigger_labels[word_idx] = clean_label

        # Map labels to tokens using the helper utility
        # We pass only the portion of tokenized_output relevant to current_sentence_raw
        # This requires careful slicing of input_ids and word_ids based on character offsets
        # For simplicity in this example, we'll try to map against the full tokenized_output
        # and rely on the `map_word_labels_to_tokens` to filter based on word_ids being valid.
        # A more robust solution would slice `tokenized_output` to `current_sentence_tokens_only`.

        # To align labels precisely, let's create a *dummy* tokenized_output for just the current sentence
        # and then map labels, and insert into the full sequence.
        tokenized_current_only = tokenizer(current_sentence_raw, add_special_tokens=False, return_offsets_mapping=True)
        current_sentence_token_labels = map_word_labels_to_tokens(
            tokenized_current_only,
            sentence_tokens,
            current_sentence_gold_trigger_labels,
            trigger_label_map,
            is_trigger_labels=True
        )

        # Now, insert these labels into the final_trigger_labels based on where current sentence is in `input_ids`
        final_trigger_labels = [-100] * len(tokenized_output['input_ids'])
        current_sentence_token_start_idx = -1
        # Find the first token index of the current sentence's content (not special tokens)
        for idx, (start_offset, end_offset) in enumerate(tokenized_output['offset_mapping']):
            if start_offset >= current_sentence_start_char_in_full_text and \
               end_offset <= current_sentence_start_char_in_full_text + len(current_sentence_raw) and \
               tokenized_output['input_ids'][idx] not in tokenizer.all_special_ids:
                current_sentence_token_start_idx = idx
                break

        if current_sentence_token_start_idx != -1:
            # Need to match the tokens correctly
            mapped_idx = 0
            for full_idx in range(current_sentence_token_start_idx, len(tokenized_output['input_ids'])):
                if mapped_idx < len(current_sentence_token_labels) and \
                   tokenized_output['input_ids'][full_idx] not in tokenizer.all_special_ids:
                    final_trigger_labels[full_idx] = current_sentence_token_labels[mapped_idx]
                    mapped_idx += 1
                elif tokenized_output['input_ids'][full_idx] == tokenizer.sep_token_id and mapped_idx >= len(current_sentence_token_labels):
                     break # Reached the SEP after current sentence

        trigger_model_inputs.append({
            'input_ids': tokenized_output['input_ids'],
            'attention_mask': tokenized_output['attention_mask'],
            'token_type_ids': token_type_ids, # Corrected segment IDs
            'labels': final_trigger_labels
        })


        # --- B. Prepare Input for Argument Model ---
        # Input format: [CLS] $type [TYPE] H_1 [SEP] ... [SEP] xt,1 ... [TRG] ... [/TRG] ... xt,|Xt| [SEP]
        # Labels: BIO-O for current sentence (excluding trigger tokens)

        if turn_key in events_data:
            for event_type_full, event_details in events_data[turn_key].items():
                base_event_type = event_type_full.split(':')[0]
                event_type_placeholder = event_type_map.get(base_event_type, "unknown_event_type")

                # Get gold triggers for this event (argument model training uses gold triggers)
                trigger_spans = [] # List of (start_word_idx, end_word_idx) for the current sentence
                for trg_str_dict in event_details.get("triggers", []):
                    try:
                        trg_dict = eval(trg_str_dict) # Exercise caution with eval()
                        indices = [int(i) for i in trg_dict['indices'].split(' ')]
                        trigger_spans.append((indices[0], indices[-1]))
                    except Exception as e:
                        print(f"Error parsing trigger info: {trg_str_dict} in {filepath} - {e}")
                        continue

                # Get argument labels for the current turn/event
                current_sentence_gold_argument_labels: List[str] = ["O"] * len(sentence_tokens)
                if event_details['labels']:
                    json_label_sequence = event_details['labels'][0]
                    for word_idx, full_label_str in enumerate(json_label_sequence):
                        clean_label = full_label_str.split(': ')[-1] if ': ' in full_label_str else full_label_str
                        # Heuristic: if it's B/I and NOT a trigger-related label
                        base_name_from_label = clean_label.split('-',1)[1] if '-' in clean_label else ""
                        if (clean_label.startswith(("B-", "I-"))) and \
                           base_name_from_label not in event_type_map:
                            current_sentence_gold_argument_labels[word_idx] = clean_label
                        elif clean_label == "O":
                            current_sentence_gold_argument_labels[word_idx] = "O"

                for trg_start_idx, trg_end_idx in trigger_spans:
                    # Construct the raw text string for the argument model's input
                    arg_full_text_parts = [tokenizer.cls_token, event_type_placeholder, "[TYPE]"]
                    arg_token_type_ids = [0, 0, 0] # For [CLS], type, [TYPE]

                    for h_text in history_raw_texts:
                        arg_full_text_parts.append(h_text)
                        arg_full_text_parts.append(tokenizer.sep_token)
                        arg_token_type_ids.extend([0] * (len(tokenizer.tokenize(h_text)) + 1))

                    # Add current sentence with [TRG] markers
                    current_sentence_with_trg_markers = []
                    for s_idx, word in enumerate(sentence_tokens):
                        if s_idx == trg_start_idx:
                            current_sentence_with_trg_markers.append("[TRG]")
                        current_sentence_with_trg_markers.append(word)
                        if s_idx == trg_end_idx:
                            current_sentence_with_trg_markers.append("[/TRG]")
                    current_sentence_raw_with_trg = " ".join(current_sentence_with_trg_markers)

                    arg_full_text_parts.append(current_sentence_raw_with_trg)
                    # Extend token_type_ids for current sentence part
                    arg_token_type_ids.extend([1] * len(tokenizer.tokenize(current_sentence_raw_with_trg)))

                    arg_full_text = " ".join(arg_full_text_parts)

                    tokenized_arg_output = tokenizer(
                        arg_full_text,
                        is_split_into_words=False,
                        padding='max_length',
                        truncation=True,
                        max_length=max_length,
                        return_offsets_mapping=True,
                        return_token_type_ids=True # Will be overwritten for clarity
                    )

                    # Manually create token_type_ids for argument model (segment 0 for meta/history, 1 for current sentence/trigger part)
                    current_sentence_start_char_in_full_arg_text = arg_full_text.find(current_sentence_raw_with_trg)
                    arg_token_type_ids_final = [0] * len(tokenized_arg_output['input_ids'])
                    if current_sentence_start_char_in_full_arg_text != -1:
                        for token_idx, (start_offset, end_offset) in enumerate(tokenized_arg_output['offset_mapping']):
                            if start_offset >= current_sentence_start_char_in_full_arg_text and \
                               end_offset <= current_sentence_start_char_in_full_arg_text + len(current_sentence_raw_with_trg):
                                arg_token_type_ids_final[token_idx] = 1

                    # Map argument labels. Trigger words should be ignored (-100).
                    final_argument_labels = [-100] * len(tokenized_arg_output['input_ids'])

                    # Get BERT tokens for the current sentence with TRG markers, for mapping
                    tokenized_current_with_trg_only = tokenizer(current_sentence_raw_with_trg, add_special_tokens=False, return_offsets_mapping=True)
                    current_sentence_arg_token_labels = map_word_labels_to_tokens(
                        tokenized_current_with_trg_only,
                        current_sentence_with_trg_markers, # Pass the list with [TRG] as words
                        current_sentence_gold_argument_labels, # This is still based on original words
                        argument_label_map,
                        is_trigger_labels=False
                    )
                    # Logic to mark [TRG] and [/TRG] tokens and original trigger words as IGNORE
                    # This requires iterating through `current_sentence_with_trg_markers` and `tokenized_current_with_trg_only.word_ids()`
                    # to correctly assign -100 for the actual trigger span and markers.

                    # Simplified: Recreate the label mapping here with precise ignore logic
                    mapped_idx = 0
                    for full_idx, (start_offset, end_offset) in enumerate(tokenized_arg_output['offset_mapping']):
                        if full_idx < len(arg_token_type_ids_final) and arg_token_type_ids_final[full_idx] == 1: # Only map if it's part of current sentence segment
                            # Map token offset to word in `current_sentence_with_trg_markers`
                            relative_start_offset = start_offset - current_sentence_start_char_in_full_arg_text
                            relative_end_offset = end_offset - current_sentence_start_char_in_full_arg_text

                            current_word_char_offset = 0
                            found_word_idx = -1
                            for word_idx, word in enumerate(current_sentence_with_trg_markers):
                                word_end_char_offset = current_word_char_offset + len(word)
                                if relative_start_offset >= current_word_char_offset and \
                                   relative_end_offset <= word_end_char_offset:
                                    found_word_idx = word_idx
                                    break
                                current_word_char_offset = word_end_char_offset + 1 # +1 for space

                            if found_word_idx != -1:
                                original_word_in_sentence = sentence_tokens[found_word_idx] if found_word_idx < len(sentence_tokens) else None

                                # Check if this token is part of the marked trigger span
                                is_token_in_trigger_span = False
                                if original_word_in_sentence and (trg_start_idx <= found_word_idx <= trg_end_idx):
                                     is_token_in_trigger_span = True

                                if current_sentence_with_trg_markers[found_word_idx] in ["[TRG]", "[/TRG]"] or is_token_in_trigger_span:
                                    final_argument_labels[full_idx] = -100 # Ignore trigger markers and actual trigger tokens
                                elif found_word_idx < len(current_sentence_gold_argument_labels):
                                    label_str = current_sentence_gold_argument_labels[found_word_idx]
                                    final_argument_labels[full_idx] = argument_label_map.get(label_str, -100)
                                else:
                                    final_argument_labels[full_idx] = -100 # Should not happen

                            else: # Likely a subword not cleanly mapped or token between words
                                final_argument_labels[full_idx] = -100
                        else: # Not part of current sentence segment
                            final_argument_labels[full_idx] = -100

                    argument_model_inputs.append({
                        'input_ids': tokenized_arg_output['input_ids'],
                        'attention_mask': tokenized_arg_output['attention_mask'],
                        'token_type_ids': arg_token_type_ids_final,
                        'labels': final_argument_labels
                    })

        # After processing the current turn, add current sentence to history
        history_raw_texts.append(current_sentence_raw)

    return trigger_model_inputs, argument_model_inputs


def load_and_process_all_data(
    data_dirs: List[str],
    tokenizer: PreTrainedTokenizerFast,
    trigger_label_map: Dict[str, int],
    argument_label_map: Dict[str, int],
    event_type_map: Dict[str, str],
    max_length: int = config.MAX_SEQ_LENGTH
) -> Tuple[EventExtractionDataset, EventExtractionDataset]:
    """
    Loads all data from specified directories, processes them, and returns
    PyTorch Dataset objects for triggers and arguments.
    """
    all_trigger_inputs = []
    all_argument_inputs = []

    for data_dir in data_dirs:
        print(f"Processing data from: {data_dir}")
        for filename in os.listdir(data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(data_dir, filename)
                trg_inputs, arg_inputs = prepare_inputs_for_model(
                    filepath, tokenizer, trigger_label_map, argument_label_map, event_type_map, max_length
                )
                all_trigger_inputs.extend(trg_inputs)
                all_argument_inputs.extend(arg_inputs)

    print(f"Total processed trigger examples: {len(all_trigger_inputs)}")
    print(f"Total processed argument examples: {len(all_argument_inputs)}")

    return EventExtractionDataset(all_trigger_inputs), EventExtractionDataset(all_argument_inputs)