"""LLM text-generation pipeline factory."""

from __future__ import annotations

import gc
import os

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

load_dotenv()

# Models that are small enough to load without quantisation
_LIGHTWEIGHT_MODELS: frozenset[str] = frozenset({"gpt2"})


def load_generator(
    model: str = "gpt2",
    token: str | None = None,
    temperature: float = 1.0,
):
    """Build a Hugging Face ``text-generation`` pipeline.

    For models not in the lightweight set the model is loaded with 4-bit
    quantisation (BitsAndBytes NF4) to keep GPU memory requirements low.

    Parameters
    ----------
    model:
        Hugging Face model identifier (name or local path).
    token:
        Hugging Face API token.  When *None* the value of the
        ``HUGGINGFACE_API_TOKEN`` environment variable is used.
    temperature:
        Sampling temperature forwarded to the pipeline.

    Returns
    -------
    transformers.Pipeline
        A ready-to-use ``text-generation`` pipeline.

    Raises
    ------
    ValueError
        If no Hugging Face token can be resolved.
    """
    hf_token = token or os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        raise ValueError(
            "HUGGINGFACE_API_TOKEN not found. "
            "Set it in your .env file or pass it as the 'token' argument."
        )

    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    if device == 0:
        torch.cuda.empty_cache()
        gc.collect()

    if model in _LIGHTWEIGHT_MODELS:
        return pipeline(
            "text-generation",
            model=model,
            token=hf_token,
            device=device,
            temperature=temperature,
        )

    print(f"Loading '{model}' with 4-bit quantisation (NF4).")
    tokenizer = AutoTokenizer.from_pretrained(model, token=hf_token)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    return pipeline(
        "text-generation",
        model=loaded_model,
        tokenizer=tokenizer,
        temperature=temperature,
    )
