from __future__ import annotations

import pickle
import string
from pathlib import Path
from typing import Any

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

from app.constants import ENGLISH_STOP_WORDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


class SpamPredictor:
    def __init__(self) -> None:
        self._tokenizer = TreebankWordTokenizer()
        self._stemmer = PorterStemmer()
        self._vectorizer = self._load_pickle(ARTIFACT_DIR / "vectorizer.pkl")
        self._model = self._load_pickle(ARTIFACT_DIR / "model.pkl")

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        if not path.is_file():
            raise RuntimeError(f"Required model artifact is missing: {path}")

        with path.open("rb") as artifact:
            return pickle.load(artifact)

    def transform_text(self, text: str) -> str:
        tokens = self._tokenizer.tokenize(text.lower())
        cleaned_tokens = (
            token
            for token in tokens
            if token.isalnum()
            and token not in ENGLISH_STOP_WORDS
            and token not in string.punctuation
        )
        return " ".join(self._stemmer.stem(token) for token in cleaned_tokens)

    def predict(self, message: str) -> bool:
        transformed_message = self.transform_text(message)
        vector = self._vectorizer.transform([transformed_message])
        return bool(self._model.predict(vector)[0])
