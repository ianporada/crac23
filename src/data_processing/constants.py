SPEAKER_START = "[SPEAKER_START]"
SPEAKER_END = "[SPEAKER_END]"

MODEL_TO_MAX_LEN = {
    "longformer": 4096,
    "spanbert": 512,
    "xlmroberta": 512,
    "mt5": 1024,
}

MODEL_TO_MODEL_STR = {
    "longformer": "allenai/longformer-large-4096",
    "spanbert": "bert-base-cased",
    "xlmroberta": "xlm-roberta-base",
    "mt5": "google/mt5-base",
}
