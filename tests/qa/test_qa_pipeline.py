from gistify.qa import qa_pipeline
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer


def test_answer():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    response = qa_pipeline.answer(
        question="Why black hole acts like an ideal black body?",
        context="""A black hole is a region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves,
        has enough energy to escape it. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to
        form a black hole. The boundary of no escape is called the event horizon. Although it has a great effect on the fate and
        circumstances of an object crossing it, it has no locally detectable features according to general relativity. In many ways,
        a black hole acts like an ideal black body, as it reflects no light.
        """,
        tokenizer=tokenizer,
        model=model,
    )

    assert len(response) > 0
