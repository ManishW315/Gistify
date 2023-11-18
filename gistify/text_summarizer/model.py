import pytorch_lightning as pl
from gistify.config import SummarizationConfig
from transformers import AdamW, BartForConditionalGeneration


class SummaryModel(pl.LightningModule):
    """Lightning model to finetune LLM."""

    def __init__(self, learning_rate):
        super().__init__()

        self.model = BartForConditionalGeneration.from_pretrained(SummarizationConfig.MODEL_NAME, return_dict=True)
        self.lr = learning_rate

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["summary_input_ids"]
        labels_attention_mask = batch["summary_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask,
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["summary_input_ids"]
        labels_attention_mask = batch["summary_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=labels_attention_mask,
        )

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer
