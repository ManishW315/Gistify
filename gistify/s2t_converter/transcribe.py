import argparse
import os
import tempfile

import torch
from gistify.config import Speech2TextConfig, logger
from gistify.s2t_converter.utils import (_return_yt_html_embed,
                                         download_yt_audio)
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read


def yt_transcribe(yt_url, task, pipe):
    html_embed_str = _return_yt_html_embed(yt_url)

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "video.mp4")
        try:
            logger.info("Downloading youtube audio.")
            download_yt_audio(yt_url, filepath)
        except Exception as e:
            logger.error(e)
        with open(filepath, "rb") as f:
            logger.info("Reading audio file.")
            inputs = f.read()

    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    logger.info("Running speech to text pipeline.")
    text = pipe(inputs, batch_size=Speech2TextConfig.BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]

    return html_embed_str, text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input url and task.")

    parser.add_argument("--url", metavar="", type=str, help="url of the YouTube video.")
    parser.add_argument(
        "--task",
        metavar="",
        type=str,
        default="transcribe",
        help="Task that needs to be performed. Parse 'transcribe' if the audio of video is in english otherwise parse 'translate' to first transcribe and then translate.",
    )

    args = parser.parse_args()

    yt_url = args.url
    task = args.task

    device = 0 if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=Speech2TextConfig.MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )

    html_embed_str, text = yt_transcribe(yt_url=yt_url, task=task, pipe=pipe)
    print(text)
