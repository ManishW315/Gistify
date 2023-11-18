import pprint

import torch
from gistify.config import Speech2TextConfig, SummarizationConfig
from gistify.s2t_converter.transcribe import yt_transcribe
from gistify.text_summarizer.model import SummaryModel
from gistify.text_summarizer.predict import summarize
from transformers import BartTokenizer, pipeline


def integrate_s2t_sum(
    yt_path,
):
    html_embed_str, text = yt_transcribe(yt_path, "transcribe", pipe)

    input_list = text.split(". ")
    generated_text = summarize(text, num_beams=5)
    output_list = [summarize(sentence, max_length=64) for sentence in input_list]

    return text, generated_text, output_list


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(width=100, indent=4)
    device = 0 if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=Speech2TextConfig.MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )
    tokenizer = BartTokenizer.from_pretrained(SummarizationConfig.MODEL_NAME)
    trained_model = SummaryModel.load_from_checkpoint(SummarizationConfig.artifacts_checkpoint_path)
    trained_model.freeze()
    text, generated_text, output_list = integrate_s2t_sum("https://www.youtube.com/shorts/BBCbIxuBJws")
    pp.pprint(text)
    pp.pprint(generated_text)
    print(output_list)
