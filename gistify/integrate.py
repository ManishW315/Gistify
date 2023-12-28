import argparse
import pprint
from typing import List, Tuple

import torch
from gistify.config import Speech2TextConfig, SummarizationConfig, logger
from gistify.qa.qa_pipeline import answer
from gistify.s2t_converter.transcribe import yt_transcribe
from gistify.text_summarizer.model import SummaryModel
from gistify.text_summarizer.predict import summarize
from transformers import BartTokenizer, pipeline


def integrate_s2t_sum(yt_path: str) -> Tuple[str, str, List]:
    """Integration of Speech to Text and Summarization pipeline.

    Args:
        yt_path (str): YouTube video to summarize.

    Returns:
        Tuple[str, str, List]: Transcribed text, Summarized Text, list of Intermediate summarized text (Each line summarized).
    """
    html_embed_str, text = yt_transcribe(yt_path, "transcribe", pipe)

    input_list = text.split(". ")
    generated_text = summarize(text, num_beams=5)
    output_list = [summarize(sentence, max_length=64) for sentence in input_list]

    return text, generated_text, output_list


def integrate_s2t_qa(yt_path: str, question: str) -> str:
    """Integration of Speech to Text and Question Answering pipeline.

    Args:
        yt_path (str): YouTube video to learn and do question answering.
        question (str): Question to ask from the video.

    Returns:
        str: Response/answer.
    """
    html_embed_str, text = yt_transcribe(yt_path, "transcribe", pipe)
    response = answer(question, text)

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input YouTube video link and task.")

    parser.add_argument("--vid", metavar="", type=str, help="Input YouTube video link")
    parser.add_argument(
        "--task", metavar="", type=int, help="Task to perform on youtube video/audio. parse 'sum' to summarize or 'qa' for question answering."
    )
    parser.add_argument(
        "--question",
        metavar="",
        type=int,
        default="null",
        help="Question to ask.",
    )

    args = parser.parse_args()
    yt_vid = args.vid
    task = args.task
    question = args.question

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

    if task == "sum":
        text, generated_text, output_list = integrate_s2t_sum("<yt_url>")
        pp.pprint(text)
        pp.pprint(generated_text)
        print(output_list)

    elif task == "qa" and question != "null":
        response = integrate_s2t_qa("<yt_url>", "<question>")

    else:
        logger.error("Arguments parsed incorrectly or are incomplete. Type `--help` for help.")
