import pytorch_lightning as pl
from gistify.config import SummarizationConfig
from gistify.text_summarizer.data import SummaryDataModule, load_data
from gistify.text_summarizer.model import SummaryModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import BartTokenizer


def train_llm(
    batch_size: int = SummarizationConfig.batch_size,
    max_epochs: int = SummarizationConfig.max_epochs,
    max_len_in: int = SummarizationConfig.max_len_input,
    max_len_out: int = SummarizationConfig.max_len_output,
    padding: bool = SummarizationConfig.padding,
    truncation: bool = SummarizationConfig.truncation,
    add_special_tokens: bool = SummarizationConfig.add_special_tokens,
    num_workers: int = SummarizationConfig.num_workers,
):
    """Train the llm for fine tuning.

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to SummarizationConfig.batch_size.
        max_epochs (int, optional): Maximum number of epochs. Defaults to SummarizationConfig.max_epochs.
        max_len_in (int, optional): Maximum input sequence length. Defaults to SummarizationConfig.max_len_input.
        max_len_out (int, optional): Maximum output sequence length. Defaults to SummarizationConfig.max_len_output.
        padding (bool, optional): Whether to do padding or not. Defaults to SummarizationConfig.padding.
        truncation (bool, optional): Whether to do truncation or not. Defaults to SummarizationConfig.truncation.
        add_special_tokens (bool, optional): Whether to add special token or not. Defaults to SummarizationConfig.add_special_tokens.
        num_workers (int, optional): Number of workers for distributed computing. Defaults to SummarizationConfig.num_workers.
    """
    # ===== Data =====
    cnn_dataset = load_data()
    tokenizer = BartTokenizer.from_pretrained(SummarizationConfig.MODEL_NAME)
    data = SummaryDataModule(cnn_dataset, tokenizer, batch_size, max_len_in, max_len_out, padding, truncation, add_special_tokens, num_workers)

    # ===== Model =====
    model = SummaryModel()

    # ===== Callback =====
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
    # train
    trainer.fit(model, data)


if __name__ == "__main__":
    train_llm()
