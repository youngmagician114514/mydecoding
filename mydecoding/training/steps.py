import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator

from mydecoding.models.dual_decoder import DualDecoderModel


def train_step(model: DualDecoderModel, batch: dict, optimizer, scaler=None):
    """
    One optimization step; intended to be plugged into a custom loop or HF Trainer.
    """
    model.train()
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)
    labels = batch["labels"]

    def forward():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output.loss

    if scaler is not None:
        with torch.cuda.amp.autocast():
            loss = forward()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss = forward()
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


def build_dataloader(dataset, tokenizer, batch_size: int = 1):
    def tokenize(example):
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = dataset.map(tokenize)
    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
