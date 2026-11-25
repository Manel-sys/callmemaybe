from llm_sdk import Small_LLM_Model
import numpy as np
import numpy.typing as npt
import json


class LLMError(Exception):
    pass


class Qwen3LLM:
    def __init__(self, device: str = "cpu") -> None:
        model_name: str = "Qwen/Qwen3-0.6B"

        try:
            self._model: Small_LLM_Model = Small_LLM_Model(
                model_name=model_name, device=device)

            self.logits_size: int = self._get_logits_size()
        except (OSError, MemoryError, RuntimeError, ValueError) as e:
            raise LLMError(f"Failed to load model '{model_name}': "
                           f"{type(e).__name__}: {e}")

        self.decoded_vocab: dict[int, str] = self._gen_decoded_vocab()
        self.allow_mask: npt.NDArray[np.bool_] = self._build_allow_mask()

    def _gen_decoded_vocab(self) -> dict[int, str]:
        try:
            vocab_path: str = self._model.get_path_to_vocab_file()

            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab: dict[str, int] = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            raise LLMError(f"Failed to read the vocabulary file: "
                           f"{type(e).__name__}: {e}")

        if not isinstance(vocab, dict):
            raise LLMError("Vocabulary file has an unexpected format: "
                           f"expected a JSON object, got "
                           f"{type(vocab).__name__}.")

        decoded_vocab: dict[int, str] = {}
        for v in vocab.values():
            decoded_vocab[v] = self._model.decode([v])

        return decoded_vocab

    def _get_logits_size(self) -> int:
        token_ids: list[int] = self._model.encode("a").tolist()[0]
        return len(self._model.get_logits_from_input_ids(token_ids))

    def _build_allow_mask(self) -> npt.NDArray[np.bool_]:
        allow_mask: npt.NDArray[np.bool_] = np.zeros(
            self.logits_size, dtype=bool)

        for token_id, token_text in self.decoded_vocab.items():
            if token_id >= self.logits_size:
                raise LLMError(f"Token id {token_id} is outside the model's"
                               f" output range ({self.logits_size})")

            if token_text != "" and "\ufffd" not in token_text:
                allow_mask[token_id] = True

        return allow_mask

    def encode(self, text: str) -> list[int]:
        try:
            tokens_list: list[int] = self._model.encode(text).tolist()[0]
        except (RuntimeError, MemoryError) as e:
            raise LLMError("Failed to encode text: "
                           f"{type(e).__name__}: {e}")

        return tokens_list

    def decode(self, ids: list[int]) -> str:
        try:
            text: str = self._model.decode(ids)
        except (OverflowError, TypeError, ValueError) as e:
            raise LLMError("Failed to decode token ids: "
                           f"{type(e).__name__}: {e}")

        return text

    def get_logits_from_input_ids(self,
                                  ids: list[int]
                                  ) -> npt.NDArray[np.float32]:
        if not ids:
            raise LLMError("Model does not accept an empty list of ids"
                           " and it must have at least one id.")

        try:
            logits_list: list[float] = (
                self._model.get_logits_from_input_ids(ids))

        except (MemoryError, RuntimeError) as e:
            raise LLMError("Failed while running the model: "
                           f"{type(e).__name__}: {e}")

        logits_array: npt.NDArray[np.float32] = np.array(
            logits_list, dtype=np.float32)

        return logits_array
