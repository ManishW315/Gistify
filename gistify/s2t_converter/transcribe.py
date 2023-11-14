import os
import tempfile

import torch
from gistify.config import Speech2TextConfig
from gistify.s2t_converter.utils import (_return_yt_html_embed,
                                         download_yt_audio)
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read


def yt_transcribe(yt_url, task, pipe):
    html_embed_str = _return_yt_html_embed(yt_url)

    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "video.mp4")
        download_yt_audio(yt_url, filepath)
        with open(filepath, "rb") as f:
            inputs = f.read()

    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    text = pipe(inputs, batch_size=Speech2TextConfig.BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]

    return html_embed_str, text


if __name__ == "__main__":
    device = 0 if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=Speech2TextConfig.MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )

    html_embed_str, text = yt_transcribe("https://www.youtube.com/shorts/BBCbIxuBJws", "transcribe", pipe)
    print(text)
