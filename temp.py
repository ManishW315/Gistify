import pprint
from typing import Dict

import pytorch_lightning as pl
from transformers import AdamW, BartForConditionalGeneration, BartTokenizer


class SummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", return_dict=True)

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
        optimizer = AdamW(self.parameters(), lr=0.0001)
        return optimizer


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
        padding="max_length",
        truncation=True,
        add_special_tokens=True,
    )
    return inputs


def summarize(text):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    text_encoding = prepare_input(tokenizer, text, 512)

    generated_ids = trained_model.model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=128,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.75,
        early_stopping=True,
    )

    preds = [tokenizer.decode(gen_id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for gen_id in generated_ids]

    return "".join(preds)


if __name__ == "__main__":
    print("Started")
    pp = pprint.PrettyPrinter(width=100, indent=4)
    print("Loading...")
    trained_model = SummaryModel.load_from_checkpoint(r"artifacts\best-checkpoint.ckpt")
    print("Done.")
    trained_model.freeze()

    input = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    input_list = input.split(".")
    print("Generating...")
    output_list = [summarize(sentence) for sentence in input_list]
    print("Done.")
    for sentence in output_list:
        pp.pprint(sentence)

    o = "".join(output_list)

    pp.pprint(summarize(o))
