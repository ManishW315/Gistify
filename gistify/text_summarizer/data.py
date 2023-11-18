from typing import Dict

import pytorch_lightning as pl
from datasets import load_dataset
from gistify.config import SummarizationConfig
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer

pl.seed_everything(42)


def load_data(
    loc: str = SummarizationConfig.DATASET_LOC,
    cfg: str = SummarizationConfig.CONFIG,
    samples: int = SummarizationConfig.SAMPLES,
    test_size: float = SummarizationConfig.test_size,
) -> Dataset:
    """Load the dataset and split it into train and test sets.

    Args:
        loc (str, optional): path to the dataset on local machine or HuggingFace data repository. Defaults to SummarizationConfig.DATASET_LOC.
        cfg (str, optional): (Specific to cnn dataset) version of the dataset. Defaults to SummarizationConfig.CONFIG.
        samples (int, optional): Number of data samples to load. Defaults to SummarizationConfig.SAMPLES.
        test_size (float, optional): proportion of test set with respect to train set. Defaults to SummarizationConfig.test_size.

    Returns:
        Dataset: Loaded and splitted dataset.
    """
    ds = load_dataset(loc, cfg, split=f"train[:{samples}]")
    ds = ds.train_test_split(test_size=test_size)
    return ds


def prepare_input(tokenizer: BartTokenizer, text: str, max_len: int, padding: bool, truncation: bool, add_special_tokens: bool) -> Dict:
    """Tokenize and prepare the input text using the provided tokenizer.

    Args:
        tokenizer (BartTokenizer): Tokenizer to use.
        text (str): input test.
        max_len (int): maximum length of the token.
        padding (bool): Whether to do padding or not.
        truncation (bool): Whether to do truncation or not.
        add_special_tokens (bool): Whether to add special tokens or not.

    Returns:
        Dict: A dictionary containing the tokenized input with keys such as 'input_ids',
            'attention_mask', etc.
    """
    inputs = tokenizer.encode_plus(
        text,
        return_tensors="pt",
        max_length=max_len,
        padding=padding,
        truncation=truncation,
        add_special_tokens=add_special_tokens,
    )
    return inputs


class SummaryDataset(Dataset):
    def __init__(
        self, data: Dataset, tokenizer: BartTokenizer, max_len_in: int, max_len_out: int, padding: bool, truncation: bool, add_special_tokens: bool
    ) -> Dict:
        """Initialize data preparation.

        Args:
            data (Dataset): raw input dataset.
            tokenizer (BartTokenizer): tokenizer to use.
            max_len_in (int): maximum input sequence length.
            max_len_out (int): maximum output/generated sequence length.
            padding (bool): Whether to do padding or not.
            truncation (bool): Whether to do truncation or not.
            add_special_tokens (bool): Whether to add special tokens or not.

        Returns:
            Dict: Dictionary containing raw input text, raw summary, input text ids, input attention mask, summary ids, summary attention mask
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len_in = max_len_in
        self.max_len_out = max_len_out
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_encoding = prepare_input(
            self.tokenizer, self.data[idx]["article"], self.max_len_in, self.padding, self.truncation, self.add_special_tokens
        )
        summary_encoding = prepare_input(
            self.tokenizer, self.data[idx]["highlights"], self.max_len_out, self.padding, self.truncation, self.add_special_tokens
        )

        return dict(
            text=self.data["article"],
            summary=self.data["highlights"],
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            summary_input_ids=summary_encoding["input_ids"].flatten(),
            summary_attention_mask=summary_encoding["attention_mask"].flatten(),
        )


class SummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: Dataset,
        tokenizer: BartTokenizer,
        batch_size: int = SummarizationConfig.batch_size,
        max_len_in: int = SummarizationConfig.max_len_input,
        max_len_out: int = SummarizationConfig.max_len_output,
        padding: bool = SummarizationConfig.padding,
        truncation: bool = SummarizationConfig.truncation,
        add_special_tokens: bool = SummarizationConfig.add_special_tokens,
        num_workers: int = SummarizationConfig.num_workers,
    ) -> None:
        """Prepare data into dataloader.

        Args:
            data (Dataset): raw input dataset.
            tokenizer (BartTokenizer): tokenizer to use.
            batch_size (int, optional): number of samples per batch of data. Defaults to SummarizationConfig.batch_size.
            max_len_in (int, optional): maximum input sequence length. Defaults to SummarizationConfig.max_len_input.
            max_len_out (int, optional): maximum output/generated sequence length.. Defaults to SummarizationConfig.max_len_output.
            padding (bool, optional): Whether to do padding or not.. Defaults to SummarizationConfig.padding.
            truncation (bool, optional): Whether to do truncation or not.. Defaults to SummarizationConfig.truncation.
            add_special_tokens (bool, optional): Whether to add special tokens or not.. Defaults to SummarizationConfig.add_special_tokens.
            num_workers (int, optional): NUmber of workers for distributed processing. Defaults to SummarizationConfig.num_workers.
        """
        super().__init__()

        self.ds = data
        self.train_ds = data["train"]
        self.val_ds = data["test"]
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len_in = max_len_in
        self.max_len_out = max_len_out
        self.padding = padding
        self.truncation = truncation
        self.add_special_tokens = add_special_tokens
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = SummaryDataset(
            self.train_ds, self.tokenizer, self.max_len_in, self.max_len_out, self.padding, self.truncation, self.add_special_tokens
        )
        self.val_dataset = SummaryDataset(
            self.val_ds, self.tokenizer, self.max_len_in, self.max_len_out, self.padding, self.truncation, self.add_special_tokens
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    cnn_dataset = load_data()
    print(cnn_dataset)
    tokenizer = BartTokenizer.from_pretrained(SummarizationConfig.MODEL_NAME)
    data = SummaryDataModule(cnn_dataset, tokenizer)
    print(type(data))
