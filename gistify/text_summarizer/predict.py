import argparse
import pprint

import torch
from gistify.config import SummarizationConfig, logger
from gistify.text_summarizer.data import prepare_input
from gistify.text_summarizer.model import SummaryModel
from transformers import BartTokenizer


def summarize(text: str, max_length: int, num_beams: int, repetition_penalty: float, length_penalty: float):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to {device}.")
    text_encoding = prepare_input(
        tokenizer,
        text,
        SummarizationConfig.max_len_input,
        SummarizationConfig.padding,
        SummarizationConfig.truncation,
        SummarizationConfig.add_special_tokens,
    ).to(device)

    logger.info("Generating summarized text.")
    generated_ids = trained_model.model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=max_length,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        early_stopping=True,
    )

    preds = [tokenizer.decode(gen_id, skip_special_tokens=True, cleanup_tokenization_spaces=True) for gen_id in generated_ids]

    return "".join(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input text and set prediction parameter values.")

    parser.add_argument("--text", metavar="", type=str, help="Text that needs to be summarized")
    parser.add_argument(
        "--max_length", metavar="", type=int, default=SummarizationConfig.max_length, help="The maximum acceptable length of input sequence."
    )
    parser.add_argument(
        "--num_beams",
        metavar="",
        type=int,
        default=SummarizationConfig.num_beams,
        help="The parameter for number of beams. If higher than 1 then effectively switches from greedy search to beam search.",
    )
    parser.add_argument(
        "--repetition_penalty",
        metavar="",
        type=float,
        default=SummarizationConfig.repetition_penalty,
        help=" The parameter for repetition penalty. A value of 1.0 means no penalty.",
    )
    parser.add_argument(
        "--length_penalty",
        metavar="",
        type=float,
        default=SummarizationConfig.length_penalty,
        help="The parameter for length penalty. length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.",
    )

    args = parser.parse_args()

    text = args.text
    max_length = args.max_length
    num_beams = args.num_beams
    repetition_penalty = args.repetition_penalty
    length_penalty = args.length_penalty

    pp = pprint.PrettyPrinter(width=100, indent=4)
    logger.info("Loading Summarization Tokenizer")
    tokenizer = BartTokenizer.from_pretrained(SummarizationConfig.MODEL_NAME)
    logger.info("Loading Summarization Model")
    trained_model = SummaryModel.load_from_checkpoint(SummarizationConfig.artifacts_checkpoint_path)
    trained_model.freeze()

    print("=" * 100)
    print("Input Sentence")
    input_list = text.split(". ")
    pp.pprint(text)
    print("=" * 100)
    print("Parameters set to:")
    print(f"max_length: {max_length}")
    print(f"num_beams: {num_beams}")
    print(f"repetition_penalty: {repetition_penalty}")
    print(f"length_penalty: {length_penalty}")
    print("=" * 100)
    logger.info("Generating output for each sentence")
    output_list = [
        summarize(text=sentence, max_length=max_length, num_beams=num_beams, repetition_penalty=repetition_penalty, length_penalty=length_penalty)
        for sentence in input_list
    ]
    for sentence in output_list:
        pp.pprint(sentence)

    print("=" * 100)
    logger.info("Generating Final output")
    pp.pprint(summarize(text=text, max_length=max_length, num_beams=num_beams, repetition_penalty=repetition_penalty, length_penalty=length_penalty))
    print("=" * 100)
