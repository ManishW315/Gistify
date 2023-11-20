from gistify.config import SummarizationConfig
from gistify.text_summarizer import data
from transformers import BartTokenizer


def test_load_data():
    samples = 500
    test_size = 0.1
    ds = data.load_data(samples=samples, test_size=test_size)
    assert (
        ds["train"].shape[0] + ds["test"].shape[0] == samples
        and test_size - 0.01 <= ds["test"].shape[0] / (ds["train"].shape[0] + ds["test"].shape[0]) <= test_size + 0.01
    )


def test_prepare_input():
    tokenizer = BartTokenizer.from_pretrained(SummarizationConfig.MODEL_NAME)
    max_len = 20
    padding = "max_length"
    truncation = True
    add_special_token = True
    text = "This is a test for prepare input function."
    tokenized_input = data.prepare_input(
        tokenizer=tokenizer, text=text, max_len=max_len, padding=padding, truncation=truncation, add_special_tokens=add_special_token
    )
    assert len(tokenized_input["input_ids"][0]) == max_len
