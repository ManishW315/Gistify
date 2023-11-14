from typing import Dict

import pytorch_lightning as pl
from datasets import load_dataset
from gistify.config import SummarizationConfig
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer

pl.seed_everything(42)


def load_data(loc=SummarizationConfig.DATASET_LOC, cfg=SummarizationConfig.CONFIG, samples=15000):
    ds = load_dataset(loc, cfg, split=f"train[:{samples}]")
    ds = ds.train_test_split(test_size=0.2)
    return ds


def prepare_input(tokenizer: BartTokenizer, text: str, max_len) -> Dict:
    """Tokenize and prepare the input text using the provided tokenizer.

    Args:
        tokenizer (RobertaTokenizer): The Roberta tokenizer to encode the input.
        text (str): The input text to be tokenized.

    Returns:
        inputs (dict): A dictionary containing the tokenized input with keys such as 'input_ids',
            'attention_mask', etc.
    """
    inputs = tokenizer.encode_plus(
        text,
        return_tensors="pt",
        max_length=max_len,
        padding=SummarizationConfig.padding,
        truncation=SummarizationConfig.truncation,
        add_special_tokens=SummarizationConfig.add_special_tokens,
    )
    return inputs


class SummaryDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_encoding = prepare_input(self.tokenizer, self.data[idx]["article"], 512)
        summary_encoding = prepare_input(self.tokenizer, self.data[idx]["highlights"], 256)

        return dict(
            text=self.data["article"],
            summary=self.data["highlights"],
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            summary_input_ids=summary_encoding["input_ids"].flatten(),
            summary_attention_mask=summary_encoding["attention_mask"].flatten(),
        )


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, data, tokenizer, batch_size):
        super().__init__()

        self.ds = data
        self.train_ds = data["train"]
        self.val_ds = data["test"]
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SummaryDataset(self.train_ds, self.tokenizer)
        self.val_dataset = SummaryDataset(self.val_ds, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=SummarizationConfig.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=SummarizationConfig.num_workers)


if __name__ == "__main__":
    cnn_dataset = load_data()
    print(cnn_dataset)
    tokenizer = BartTokenizer.from_pretrained(SummarizationConfig.MODEL_NAME)
    data = SummaryDataModule(cnn_dataset, tokenizer, SummarizationConfig.batch_size)
    print(data)
