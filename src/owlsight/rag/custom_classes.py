from enum import Enum
from pathlib import Path
from typing import Any, Optional, Type
import pickle
import faiss
import numpy as np

from pydantic import BaseModel


class SearchMethod(str, Enum):
    """Supported search methods."""

    TFIDF = "tfidf"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    HASHING = "hashing"


class SearchResult(BaseModel):
    """Model to store essential search results with type validation."""

    document: str
    document_name: str
    score: float
    method: Optional[str] = None
    weighted_score: Optional[float] = None


class CacheMixin:
    """Mixin class for caching functionality with optional FAISS support."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
        faiss_mode: bool = False,
        faiss_class: Optional[Type[faiss.Index]] = None,
    ):
        """Initialize the cache mixin.

        Parameters:
        ----------
        cache_dir:
            Directory for caching
        cache_dir_suffix:
            Suffix for cache files
        faiss_mode:
            If True, uses FAISS for vector storage instead of pickle files
        faiss_method:
            FAISS method class to be used for indexing
        """
        if cache_dir and not cache_dir_suffix:
            raise ValueError("cache_dir_suffix must be provided when cache_dir is specified")

        # we check if faiss_class is a legitimate class in the faiss module
        if faiss_class and not issubclass(faiss_class, faiss.Index):
            raise ValueError("faiss_class must be a subclass of faiss.Index")

        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_dir_suffix = cache_dir_suffix
        self.faiss_mode = faiss_mode
        self.faiss_class = faiss_class if faiss_class else faiss.IndexFlatIP

        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get_suffix_filename(self) -> str:
        """Get the suffix filename."""
        return self.cache_dir_suffix if self.cache_dir_suffix else ""

    def get_full_cache_path(self) -> Path:
        """Get full cache path."""
        if not self.cache_dir:
            raise ValueError("Cache directory not provided")
        extension = ".index" if self.faiss_mode else ".pkl"
        return self.cache_dir / f"{self.get_suffix_filename()}{extension}"

    def save_data(self, data: Any) -> Optional[faiss.Index]:
        """Save data to cache.

        For FAISS mode, data should be a numpy array of vectors.
        For regular mode, data can be any pickle-able object.
        """
        if not self.cache_dir:
            return

        cache_path = self.get_full_cache_path()
        if self.faiss_mode:
            if not isinstance(data, np.ndarray):
                raise ValueError("Data must be a numpy array for FAISS mode")

            return self._create_faiss_index(data, cache_path)
        else:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

    def load_data(self) -> Optional[faiss.Index]:
        """Load data from cache.

        Returns:
            For FAISS mode: returns the FAISS index
            For regular mode: returns the pickled object
            Returns None if no cache exists
        """
        if not self.cache_dir:
            return None

        cache_path = self.get_full_cache_path()
        if not cache_path.exists():
            return None

        if self.faiss_mode:
            index = faiss.read_index(str(cache_path))
            return index
        else:
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    def process_data_for_faiss(self, data: np.ndarray) -> None:
        """Process data for FAISS indexing."""
        if not self.faiss_mode:
            return

        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array for FAISS mode")

        if self.faiss_class == faiss.IndexFlatIP:
            data = data.reshape(1, -1)
            faiss.normalize_L2(data)  # Inner Product (IP) for cosine similarity

    def _create_faiss_index(self, data: np.ndarray, cache_path: Path) -> faiss.Index:
        dimension = data.shape[1] if len(data.shape) > 1 else data.shape[0]
        index = self.faiss_class(dimension)
        self.process_data_for_faiss(data)
        index.add(data)
        faiss.write_index(index, str(cache_path))
        return index
