import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer


def answer(question: str, text: str) -> str:
    """Question answering pipeline.

    Args:
        question (str): Question to ask from the context.
        text (str): The context from which questions are to be asked.

    Returns:
        str: response/answer.
    """
    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits)

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    response = tokenizer.decode(predict_answer_tokens)

    return response


if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    question, text = (
        "What are some limitations of JavaScript?",
        "JavaScript doesn't allow you to specify what type something is. I'm guessing that wasn't deemed very useful for simply animating some HTML buttons. It also makes sense because it's quite common in web development that variables hold different types of data depending on what the user does or what kind of data the server returns. However, if you want to develop a full-fledged web application, not having these types is a recipe for disaster. And this is also why TypeScript has become so popular. TypeScript is a superset of JavaScript that adds static types to the language. With TypeScript, you can write type annotations and the TypeScript compiler will check the types at compile time when you compile the code to JavaScript, helping you catch common errors before they run.",
    )
    response = answer(question, text)
    print(response)
