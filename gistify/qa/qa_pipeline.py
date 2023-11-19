import argparse

import torch
from gistify.config import logger
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer


def answer(question: str, context: str) -> str:
    """Question answering pipeline.

    Args:
        question (str): Question to ask from the context.
        context (str): The context from which questions are to be asked.

    Returns:
        str: response/answer.
    """
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        logger.info("Running question answering pipeline.")
        outputs = model(**inputs)

    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits)

    try:
        logger.info("Decoding qa response.")
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        response = tokenizer.decode(predict_answer_tokens)
    except Exception as e:
        logger.error(e)

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input question and context based on which the answer should be given.")

    parser.add_argument("--question", metavar="", type=str, help="Question to ask.")
    parser.add_argument("--context", metavar="", type=str, default="transcribe", help="Context based on which answer should be given.")

    args = parser.parse_args()

    question = args.question
    context = args.context

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    response = answer(question, context)
    print(response)
