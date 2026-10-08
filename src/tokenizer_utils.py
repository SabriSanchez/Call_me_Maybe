from typing import cast, List, Dict, Any
from llm_sdk import Small_LLM_Model  # type: ignore


class TokenizerUtils:
    """Utility methods for token encoding and decoding."""

    def __init__(self, model: "Small_LLM_Model") -> None:
        """Initialize the tokenizer utility."""
        self.model = model

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs.
        Args:
            text: Input text.
        Returns:
            List of token IDs.
        """
        return cast(List[int], self.model.encode(text).tolist()[0])

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs into text.
        Args:
            token_ids: Token IDs to decode.
        Returns:
            Decoded text.
        """
        return cast(str, self.model.decode(token_ids))

    def encode_functions(
        self,
        functions: List[Dict[str, Any]]
    ) -> Dict[str, List[int]]:
        """Encode all function names.
        Args:
            functions: Function definitions.
        Returns:
            Mapping from function names to token IDs.
        """
        return {
            str(fn["name"]): self.encode(str(fn["name"]))
            for fn in functions
        }
