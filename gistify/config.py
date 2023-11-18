from dataclasses import dataclass


@dataclass
class Speech2TextConfig:
    """Configurations for speech to text conversion."""

    MODEL_NAME = "openai/whisper-large-v2"
    BATCH_SIZE = 8
    FILE_LIMIT_MB = 1000
    YT_LENGTH_LIMIT_S = 3600


@dataclass
class SummarizationConfig:
    """Configurations for text summarization."""

    DATASET_LOC = "cnn_dailymail"
    SAMPLES = 15000
    test_size = 0.2
    CONFIG = "3.0.0"
    MODEL_NAME = "facebook/bart-base"
    max_len_input = 512
    max_len_output = 128
    padding = "max_length"
    truncation = True
    add_special_tokens = True
    batch_size = 8
    num_workers = 2
    learning_rate = 0.0001
    max_epochs = 1
