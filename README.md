# Gistify

Gistify is a tool/script designed to simplify the process of extracting audio from YouTube videos, transcribing the content, and providing valuable insights through summarization or question-answer generation using state-of-the-art transformer models.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Docs](#docs)
- [Project Structure](#project-structure)
- [Usage](#usage)

## Features

- **YouTube Link Processing:** Gistify seamlessly takes a YouTube link as input extract and transcribes the audio into text using advanced speech-to-text technology.

- **Summarization:** Gistify can generate concise and informative summaries of the transcribed content, making it easier for users to grasp the key points of the video.

- **Question-Answer Generation:** Engage with the content interactively by prompting Gistify to answer specific questions related to the transcribed text, leveraging powerful transformer models.

## Installation

```bash
# Clone the repository
git clone https://github.com/ManishW315/Gistify.git

# Navigate to the project directory
cd Gistify

# Create a conda environment
conda create -n venv python=3.11 -y

# Activate the environment
conda activate venv

# Install gistify package
pip install -e .
```

## Docs
To understand the working and training process, check the documentation [here](https://manishw315.github.io/Gistify/).

## Project Structure
The project is organized as follows:

*Core project files:*

<pre>
Gistify
│
└───gistify
    │   config.py
    │   integrate.py
    │   __init__.py
    │
    ├───qa
    │       qa_pipeline.py
    │       __init__.py
    │
    ├───s2t_converter
    │       transcribe.py
    │       utils.py
    │       __init__.py
    │
    └───text_summarizer
            data.py
            model.py
            predict.py
            train.py
            __init__.py
</pre>

## Usage
Choose to either summarize or ask question about the video.

```bash
# Perform summarization on video
python gistify\integrate.py --vid "https://www.youtube.com/example" --task "sum"

# Perform question-answering on video
python gistify\integrate.py --vid "https://www.youtube.com/example" --task "qa" --question "Ask question here"
```

Gistify also has the handy feature to summarize or generate question-answer for a normal input text (without any video to audio to text transformation)

```bash
# Perform summarization on text [Type `--help` to see additional arguments]
python gistify\text_summarizer\predict.py --text "Input text here"

# Perform question-answering on text
python gistify\qa\qa_pipeline.py --context "Input text here" --question "Ask question here"
```

Check the documentation and `config.py` file for own custom training.

The training script requires us to have a [Weights and Biases](https://wandb.ai/site) account for experiment tracking and visualization.

