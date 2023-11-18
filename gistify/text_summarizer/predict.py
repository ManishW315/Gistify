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
    input = """In astronomy, dark matter is a hypothetical form of matter that appears to not interact with light or the electromagnetic field. Dark matter is implied by gravitational effects which cannot be explained by general relativity unless more matter is present than can be seen, which include: formation and evolution of galaxies, gravitational lensing, observable universe's current structure, mass position in galactic collisions, motion of galaxies within galaxy clusters, and cosmic microwave background anisotropies.
In the standard Lambda-CDM model of cosmology, the mass–energy content of the universe is 5% ordinary matter, 26.8% dark matter, and 68.2% a form of energy known as dark energy. Thus, dark matter constitutes 85% of the total mass, while dark energy and dark matter constitute 95% of the total mass–energy content.
Dark matter is not known to interact with ordinary baryonic matter and radiation except through gravity, making it difficult to detect in the laboratory. The leading explanation is that dark matter is some as-yet-undiscovered subatomic particle, such as weakly interacting massive particles (WIMPs) or axions. The other main possibility is that dark matter is composed of primordial black holes.
Dark matter is classified as "cold", "warm", or "hot" according to its velocity (more precisely, its free streaming length). Recent models have favored a cold dark matter scenario, in which structures emerge by the gradual accumulation of particles, but after a half century of fruitless dark matter particle searches, more recent gravitational wave and James Webb Space Telescope observations have considerably strengthened the case for primordial and direct collapse black holes.
Although the astrophysics community generally accepts dark matter's existence, a minority of astrophysicists, intrigued by specific observations that are not well-explained by ordinary dark matter, argue for various modifications of the standard laws of general relativity. These include modified Newtonian dynamics, tensor–vector–scalar gravity, or entropic gravity. So far none of the proposed modified gravity theories can successfully describe every piece of observational evidence at the same time, suggesting that even if gravity has to be modified, some form of dark matter will still be required."""
    input_list = input.split(". ")
    pp.pprint(input)
    print("=" * 100)
    print("Generating output for each sentence")
    output_list = [summarize(sentence, max_length=64) for sentence in input_list]
    for sentence in output_list:
        pp.pprint(sentence)

    print("=" * 100)
    print("Generating Final output")
    pp.pprint(summarize(input, num_beams=5))
    print("=" * 100)
