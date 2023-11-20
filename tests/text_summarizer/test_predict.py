import pytest

import torch
from gistify.config import SummarizationConfig
from gistify.text_summarizer import predict
from gistify.text_summarizer.model import SummaryModel
from transformers import BartTokenizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Faster prediction requires a GPU.")
def test_summarize():
    tokenizer = BartTokenizer.from_pretrained(SummarizationConfig.MODEL_NAME)
    trained_model = SummaryModel.load_from_checkpoint(SummarizationConfig.artifacts_checkpoint_path)
    trained_model.freeze()
    text = """A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves,
        has enough energy to escape it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to
        form a black hole. The boundary of no escape is called the event horizon. Although it has a great effect on the fate and
        circumstances of an object crossing it, it has no locally detectable features according to general relativity. In many ways,
        a black hole acts like an ideal black body, as it reflects no light.
        """
    output = predict.summarize(
        text=text,
        max_length=128,
        num_beams=5,
        repetition_penalty=2.5,
        length_penalty=1.75,
        tokenizer=tokenizer,
        trained_model=trained_model,
    )

    assert len(output) < len(text)
