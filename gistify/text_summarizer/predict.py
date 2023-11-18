import pprint

import torch
from gistify.config import SummarizationConfig
from gistify.text_summarizer.data import prepare_input
from gistify.text_summarizer.model import SummaryModel
from transformers import BartTokenizer


def summarize(text: str, max_length: int = 128, num_beams: int = 3, repetition_penalty: float = 2.5, length_penalty: float = 1.75):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    text_encoding = prepare_input(
        tokenizer,
        text,
        SummarizationConfig.max_len_input,
        SummarizationConfig.padding,
        SummarizationConfig.truncation,
        SummarizationConfig.add_special_tokens,
    ).to(device)

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
    pp = pprint.PrettyPrinter(width=100, indent=4)
    print("Loading Tokenizer")
    tokenizer = BartTokenizer.from_pretrained(SummarizationConfig.MODEL_NAME)
    print("Loading Model")
    trained_model = SummaryModel.load_from_checkpoint(SummarizationConfig.artifacts_checkpoint_path)
    trained_model.freeze()

    print("=" * 100)
    print("Input Sentence")
    input_text = """JavaScript doesn't allow you to specify what type something is. I'm guessing that wasn't deemed very useful for simply animating some HTML buttons. It also makes sense because it's quite common in web development that variables hold different types of data depending on what the user does or what kind of data the server returns. However, if you want to develop a full-fledged web application, not having these types is a recipe for disaster. And this is also why TypeScript has become so popular. TypeScript is a superset of JavaScript that adds static types to the language. With TypeScript, you can write type annotations and the TypeScript compiler will check the types at compile time when you compile the code to JavaScript, helping you catch common errors before they run."""
    input_list = input_text.split(". ")
    pp.pprint(input_text)
    print("=" * 100)
    print("Generating output for each sentence")
    output_list = [summarize(sentence, max_length=64) for sentence in input_list]
    for sentence in output_list:
        pp.pprint(sentence)

    print("=" * 100)
    print("Generating Final output")
    pp.pprint(summarize(input_text, num_beams=5))
    print("=" * 100)
