from typing import Dict, List, Optional
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, pre_tokenizers, processors, models


class MambaTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    SPECIAL_TOKENS = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
    }

    DNA_TOKENS = ["A", "T", "G", "C", "U", "N"]
    PROTEIN_TOKENS = [
        "D",
        "E",
        "F",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "V",
        "W",
        "Y",
    ]
    SPECIAL_CHARS = ["*", "-", "X", "B", "Z", "J"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        model_max_length: int = 12000,
        mode: str = "dna",  # "dna" or "protein" or "both"
        **kwargs,
    ):
        # Override special tokens with provided values or defaults
        for key, default in self.SPECIAL_TOKENS.items():
            if key not in kwargs:
                kwargs[key] = default

        self.mode = mode
        self.model_max_length = model_max_length

        base_tokenizer = None if tokenizer_file else self._create_tokenizer()
        super().__init__(
            tokenizer_object=base_tokenizer,
            tokenizer_file=tokenizer_file,
            model_max_length=model_max_length,
            **kwargs,
        )

    @classmethod
    def _create_vocab(cls, mode: str) -> Dict[str, int]:
        """Create vocabulary based on mode."""
        vocab = {}
        current_id = 0

        # Add special tokens first
        for token in cls.SPECIAL_TOKENS.values():
            vocab[token] = current_id
            current_id += 1

        # Add tokens based on mode
        if mode in ["dna", "both"]:
            for token in cls.DNA_TOKENS:
                vocab[token] = current_id
                current_id += 1

        if mode in ["protein", "both"]:
            for token in cls.PROTEIN_TOKENS:
                if token not in vocab:  # Avoid duplicates like 'N'
                    vocab[token] = current_id
                    current_id += 1

        # Add special characters
        # for token in cls.SPECIAL_CHARS:
        #     vocab[token] = current_id
        #     current_id += 1

        return vocab

    def _create_tokenizer(self) -> Tokenizer:
        """Create a new tokenizer instance."""
        vocab = self._create_vocab(self.mode)

        tokenizer = Tokenizer(
            models.WordLevel(vocab, unk_token=self.SPECIAL_TOKENS["unk_token"])
        )

        # Set pre-tokenizer to split into individual characters
        tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")

        # Set post-processor for adding special tokens
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.SPECIAL_TOKENS['cls_token']} $A {self.SPECIAL_TOKENS['sep_token']}",
            pair=None,
            special_tokens=[
                (
                    self.SPECIAL_TOKENS["cls_token"],
                    vocab[self.SPECIAL_TOKENS["cls_token"]],
                ),
                (
                    self.SPECIAL_TOKENS["sep_token"],
                    vocab[self.SPECIAL_TOKENS["sep_token"]],
                ),
            ],
        )

        return tokenizer

    def get_vocab_size(self) -> int:
        """Return the size of vocabulary."""
        return len(self.get_vocab())

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of token ids to a string.
        """
        text = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        # Remove spaces between characters for biological sequences
        return "".join(text.split())
