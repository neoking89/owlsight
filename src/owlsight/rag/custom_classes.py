from enum import Enum
from pathlib import Path
from typing import Any, Optional
import pickle
from dataclasses import dataclass


class SearchMethod(str, Enum):
    """Supported search methods."""

    TFIDF = "tfidf"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    HASHING = "hashing"

@dataclass
class SearchResult:
    """Model to store essential search results with type validation."""

    document: str
    document_name: str
    score: float
    method: Optional[str] = None
    weighted_score: Optional[float] = None


class CacheMixin:
    """Mixin class for caching functionality."""

    def __init__(self, cache_dir: Optional[str] = None, cache_dir_suffix: Optional[str] = None):
        """Initialize the cache mixin."""
        if cache_dir and not cache_dir_suffix:
            raise ValueError("cache_dir_suffix must be provided when cache_dir is specified")

        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_dir_suffix = cache_dir_suffix

        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get_suffix_filename(self) -> str:
        """Get the suffix filename."""
        return self.cache_dir_suffix if self.cache_dir_suffix else ""

    def get_full_cache_path(self) -> Path:
        """Get full cache path."""
        if not self.cache_dir:
            raise ValueError("Cache directory not provided")
        return self.cache_dir / f"{self.get_suffix_filename()}.pkl"

    def save_data(self, data: Any):
        """Save data to cache."""
        if self.cache_dir:
            cache_path = self.get_full_cache_path()
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

    def load_data(self) -> Optional[Any]:
        """Load data from cache."""
        if self.cache_dir:
            cache_path = self.get_full_cache_path()
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
        return None
