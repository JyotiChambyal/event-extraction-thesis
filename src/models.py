# src/models.py

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedTokenizerFast
from torchcrf import CRF # You'll need to install this: pip install pytorch-crf
from typing import Optional, Dict

import src.config as config
from .utils import WeightedCELoss # Or import FocalLoss if enabled

class HistoryAwareModel(nn.Module):
    """
    Base class for Trigger and Argument models with a shared BERT encoder and history awareness.
    """
    def __init__(self, num_labels: int, tokenizer: PreTrainedTokenizerFast):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        # Resize token embeddings to include new special tokens
        self.bert.resize_token_embeddings(len(tokenizer))

        self.num_labels = num_labels
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self.loss_fct = self._get_loss_function()

    def _get_loss_function(self):
        """Initializes the loss function based on config."""
        if config.CLASS_WEIGHTS_ENABLED:
            # You'll need to calculate class weights from your dataset during data loading
            # and pass them to the loss function. This is a placeholder.
            # Example: class_weights = torch.tensor([...], dtype=torch.float)
            # For now, it assumes weights are handled externally or default to None.
            # A more robust way is to pass class_weights to the forward method or init.
            # For simplicity, we'll assume weights are managed higher up or derived if not passed explicitly.
            # In training loop, you'd calculate and pass weights.
            return WeightedCELoss(ignore_index=-100) # Weights will be set dynamically
        elif config.FOCAL_LOSS_ENABLED:
            # return FocalLoss(gamma=config.FOCAL_LOSS_GAMMA, ignore_index=-100)
            raise NotImplementedError("Focal Loss not implemented. Please add its definition to utils.py")
        else:
            return CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                class_weights: Optional[torch.Tensor] = None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids # Handles segment IDs
        )
        sequence_output = outputs[0] # (batch_size, sequence_length, hidden_size)
        sequence_output = self.dropout(sequence_output)

        # Classifier outputs logits for each token for each label
        logits = self.classifier(sequence_output) # (batch_size, sequence_length, num_labels)

        if labels is not None:
            # Calculate loss using CRF
            # The CRF layer expects shape (batch_size, sequence_length, num_labels)
            # The mask should be bool tensor (False for padded tokens)
            mask = (attention_mask == 1)

            # Apply class weights to the loss calculation
            if config.CLASS_WEIGHTS_ENABLED and class_weights is not None:
                self.loss_fct.weight = class_weights.to(logits.device)

            # CRF returns negative log likelihood, so we negate it for standard loss minimization
            loss = -self.crf(emissions=logits, tags=labels, mask=mask, reduction='mean')
            return loss, logits
        else:
            # For inference, decode with CRF
            # CRF decode expects shape (batch_size, sequence_length, num_labels)
            # The mask should be bool tensor (False for padded tokens)
            mask = (attention_mask == 1)
            predictions = self.crf.decode(emissions=logits, mask=mask) # List of lists of tag IDs
            return predictions, logits


class HistoryAwareTriggerModel(HistoryAwareModel):
    """
    Model specifically for trigger extraction.
    """
    def __init__(self, num_trigger_labels: int, tokenizer: PreTrainedTokenizerFast):
        super().__init__(num_trigger_labels, tokenizer)

class HistoryAwareArgumentModel(HistoryAwareModel):
    """
    Model specifically for argument extraction.
    """
    def __init__(self, num_argument_labels: int, tokenizer: PreTrainedTokenizerFast):
        super().__init__(num_argument_labels, tokenizer)