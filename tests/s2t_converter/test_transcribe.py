import pytest

import torch
from gistify.config import Speech2TextConfig
from gistify.s2t_converter import transcribe
from transformers import pipeline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Faster prediction requires a GPU.")
def test_yt_transcribe():
    yt_url = "https://www.youtube.com/shorts/KMZDLfmFq78"
    task = "transcribe"

    device = 0 if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=Speech2TextConfig.MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )
    transcribe.yt_transcribe(yt_url, task, pipe)
