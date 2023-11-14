from dataclasses import dataclass


@dataclass
class Speech2TextConfig:
    MODEL_NAME = "openai/whisper-large-v2"
    BATCH_SIZE = 8
    FILE_LIMIT_MB = 1000
    YT_LENGTH_LIMIT_S = 3600


@dataclass
class SummarizationConfig:
    DATASET_LOC = "cnn_dailymail"
    CONFIG = "3.0.0"
    MODEL_NAME = "facebook/bart-base"
    padding = "max_length"
    truncation = True
    add_special_tokens = True
    batch_size = 8
    num_workers = 2
