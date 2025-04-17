# model/register.py

from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
)

from .model import (
    StripedMambaConfig,
    StripedMambaForMaskedLM,
    StripedMambaForSequenceClassification,
    StripedMambaForTokenClassification,
)

from .tokenizer import MambaTokenizerFast


def register_mamba_models():
    """Register all Mamba models with the transformers library."""
    # Config
    if "striped_mamba" not in CONFIG_MAPPING._extra_content:
        CONFIG_MAPPING.register("striped_mamba", StripedMambaConfig)

    # MLM
    if StripedMambaConfig not in MODEL_FOR_MASKED_LM_MAPPING._extra_content:
        MODEL_FOR_MASKED_LM_MAPPING.register(
            StripedMambaConfig, StripedMambaForMaskedLM
        )

    # Sequence Classification
    if (
        StripedMambaConfig
        not in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING._extra_content
    ):
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.register(
            StripedMambaConfig, StripedMambaForSequenceClassification
        )

    # Token Classification
    if StripedMambaConfig not in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING._extra_content:
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.register(
            StripedMambaConfig, StripedMambaForTokenClassification
        )

    # Tokenizer
    if StripedMambaConfig not in TOKENIZER_MAPPING._extra_content:
        TOKENIZER_MAPPING.register(StripedMambaConfig, (None, MambaTokenizerFast))


__all__ = ["register_mamba_models"]
