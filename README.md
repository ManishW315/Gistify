# Video-Audio-Summarizer

## Table of Contents
- [Docs](#docs)
- [Project Structure](#project-structure)


## Docs
**See docs for here: [Docs](https://manishw315.github.io/Gistify/)**

---

## Project Structure
The project is organized as follows:

*Core project files:*


<pre>
Gistify
│
├───gistify
│   │   config.py
│   │   integrate.py
│   │   __init__.py
│   │
│   ├───qa
│   │       qa_pipeline.py
│   │       __init__.py
│   │
│   ├───s2t_converter
│   │       transcribe.py
│   │       utils.py
│   │       __init__.py
│   │
│   └───text_summarizer
│           data.py
│           model.py
│           predict.py
│           train.py
│           __init__.py
│
├───notebooks
│       text_summarization_bart_cnn.ipynb
│       whisper_speech2text_yt.ipynb
│
│
└───tests
    ├───qa
    │       test_qa_pipeline.py
    │
    ├───s2t_converter
    │       test_transcribe.py
    │
    │
    └───text_summarizer
            test_data.py
            test_predict.py
</pre>  
