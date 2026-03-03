import os
from functools import lru_cache

# ✅ IMPORTANT: import the pipeline class
from app.ml.model import CFUPipeline


@lru_cache()
def get_pipeline():
    """
    Returns a singleton CFU detection pipeline instance.
    """

    model_path = os.getenv(
        "MODEL_PATH",
        "models/cfu_detector.pt",
    )

    return CFUPipeline(model_path=model_path)