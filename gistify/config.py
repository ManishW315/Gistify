import logging
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler


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
    artifacts_checkpoint_path = os.path.join((Path(__file__).parent.parent), "artifacts", "best-checkpoint.ckpt")
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

    max_length = 128
    num_beams = 3
    repetition_penalty = 2.5
    length_penalty = 2.0


logs_path = os.path.join(Path(__file__).parent.parent, "logs")

# Create logs folder
os.makedirs(logs_path, exist_ok=True)

# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
console_handler = RichHandler(markup=True)
console_handler.setLevel(logging.INFO)

info_handler = RotatingFileHandler(
    filename=Path(logs_path, "info.log"),
    maxBytes=10485760,
    backupCount=10,
)
info_handler.setLevel(logging.INFO)

error_handler = RotatingFileHandler(
    filename=Path(logs_path, "error.log"),
    maxBytes=10485760,
    backupCount=10,
)
error_handler.setLevel(logging.ERROR)

# Create formatters
minimal_formatter = logging.Formatter(fmt="%(message)s")
detailed_formatter = logging.Formatter(fmt="%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n")

console_handler.setFormatter(fmt=minimal_formatter)
info_handler.setFormatter(fmt=detailed_formatter)
error_handler.setFormatter(fmt=detailed_formatter)
logger.addHandler(hdlr=console_handler)
logger.addHandler(hdlr=info_handler)
logger.addHandler(hdlr=error_handler)
