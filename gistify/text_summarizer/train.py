import argparse

import yaml

import pytorch_lightning as pl
from gistify.config import SummarizationConfig, logger
from gistify.text_summarizer.data import SummaryDataModule, load_data
from gistify.text_summarizer.model import SummaryModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import BartTokenizer


def train_llm(
    samples: int,
    test_size: float,
    batch_size: int,
    max_epochs: int,
    max_len_in: int,
    max_len_out: int,
    padding: bool,
    truncation: bool,
    add_special_tokens: bool,
    num_workers: int,
):
    """Train the llm for fine tuning.

    Args:
        samples (int): Number of samples to take from the data.
        test_size (float): Test set split proportion.
        batch_size (int): Number of samples per batch. Defaults to SummarizationConfig.batch_size.
        max_epochs (int): Maximum number of epochs. Defaults to SummarizationConfig.max_epochs.
        max_len_in (int): Maximum input sequence length. Defaults to SummarizationConfig.max_len_input.
        max_len_out (int): Maximum output sequence length. Defaults to SummarizationConfig.max_len_output.
        padding (bool): Whether to do padding or not. Defaults to SummarizationConfig.padding.
        truncation (bool): Whether to do truncation or not. Defaults to SummarizationConfig.truncation.
        add_special_tokens (bool): Whether to add special token or not. Defaults to SummarizationConfig.add_special_tokens.
        num_workers (int): Number of workers for distributed computing. Defaults to SummarizationConfig.num_workers.
    """
    try:
        # ===== Data =====
        cnn_dataset = load_data(samples=samples, test_size=test_size)
        tokenizer = BartTokenizer.from_pretrained(SummarizationConfig.MODEL_NAME)
        logger.info("Preparing data.")
        data = SummaryDataModule(cnn_dataset, tokenizer, batch_size, max_len_in, max_len_out, padding, truncation, add_special_tokens, num_workers)

        # ===== Model =====
        logger.info("Initializing summarization model.")
        model = SummaryModel()

        # ===== Callback =====
        logger.info("Initializing callbacks.")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath="artifacts",
            filename="best-checkpoint",
            save_top_k=1,
            verbose=True,
        )

        # ===== Experiment Tracking (WandB) =====
        wandb_logger = WandbLogger(project="Text_Summarization-bart-cnn")

        # ===== Trainer =====
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=checkpoint_callback,
            max_epochs=max_epochs,
            # accelerator="gpu",
            # devices=1,
        )
    except Exception as e:
        logger.error(e)

    # train
    try:
        logger.info("Starting training.")
        trainer.fit(model, data)
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set hyperparameters for training.")

    parser.add_argument("--config", metavar="", type=str, default="None", help="Path to the YAML configuration file.")
    parser.add_argument("--samples", metavar="", type=str, default=SummarizationConfig.SAMPLES, help="Number of samples to take from the data.")
    parser.add_argument("--test-size", metavar="", type=str, default=SummarizationConfig.test_size, help="Test set split proportion.")
    parser.add_argument("--batch_size", metavar="", type=int, default=SummarizationConfig.batch_size, help="NUmber of samples per batch.")
    parser.add_argument("--max_epochs", metavar="", type=int, default=SummarizationConfig.max_epochs, help="Number of complete data pass.")
    parser.add_argument(
        "--max_len_in", metavar="", type=int, default=SummarizationConfig.max_len_input, help="Maximum acceptable input sequence length."
    )
    parser.add_argument(
        "--max_len_out", metavar="", type=int, default=SummarizationConfig.max_len_output, help="Maximum acceptable output sequence length."
    )
    parser.add_argument("--padding", metavar="", type=bool, default=SummarizationConfig.padding, help="Whether to do padding on input sequence.")
    parser.add_argument(
        "--truncation", metavar="", type=bool, default=SummarizationConfig.truncation, help="Whether to truncate the input sequence to max_length."
    )
    parser.add_argument(
        "--add_special_tokens", metavar="", type=bool, default=SummarizationConfig.add_special_tokens, help="Whether to add special tokens or not."
    )
    parser.add_argument(
        "--num_workers", metavar="", type=int, default=SummarizationConfig.num_workers, help="NUmber of workers for distributed processing."
    )

    args = parser.parse_args()

    if args.config != "None":
        with open(args.config, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

        try:
            samples = config["data"]["samples"]
            test_size = config["data"]["test_size"]
            batch_size = config["data"]["batch_size"]
            max_len_in = config["tokenizer"]["max_len_in"]
            max_len_out = config["tokenizer"]["max_len_out"]
            padding = config["tokenizer"]["padding"]
            truncation = config["tokenizer"]["truncation"]
            add_special_tokens = config["tokenizer"]["add_special_tokens"]
            max_epochs = config["trainer"]["max_epochs"]
            num_workers = config["trainer"]["num_workers"]

        except Exception as e:
            logger.error(e)

    elif args.config == "None":
        samples = args.samples
        test_size = args.test_size
        batch_size = args.batch_size
        max_len_in = args.max_len_in
        max_len_out = args.max_len_out
        padding = args.padding
        truncation = args.truncation
        add_special_tokens = args.add_special_tokens
        max_epochs = args.max_epochs
        num_workers = args.num_workers

    train_llm(
        samples=samples,
        test_size=test_size,
        batch_size=batch_size,
        max_epochs=max_epochs,
        max_len_in=max_len_in,
        max_len_out=max_len_out,
        padding=padding,
        truncation=truncation,
        add_special_tokens=add_special_tokens,
        num_workers=num_workers,
    )
