import sys

sys.path.append("src")
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import pickle
import importlib
import inspect
import pkgutil
from abc import ABC, abstractmethod

from pydantic import BaseModel
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from owlsight.utils.deep_learning import get_best_device
from owlsight.utils.helper_functions import check_invalid_input_parameters
from owlsight.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class SearchMethod(str, Enum):
    """Supported search methods."""

    TFIDF = "tfidf"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    HASHING = "hashing"


class SearchResult(BaseModel):
    """Model to store search results with type validation."""

    document: str
    score: float
    method: Optional[str] = None
    weighted_score: Optional[float] = None
    aggregated_score: Optional[float] = None


class DocumentProcessor:
    """Handles document preprocessing and validation."""

    @staticmethod
    def process_documents(documents: List[str]) -> List[str]:
        """Process and validate input documents."""
        processed_docs = []

        for doc in documents:  # use set to remove duplicate str
            if isinstance(doc, str):
                processed_docs.append(doc)
            elif hasattr(doc, "__doc__") and doc.__doc__:
                processed_docs.append(doc.__doc__)

        if not processed_docs:
            raise ValueError("No valid documents found after processing")
        return processed_docs


class CacheMixin:
    """Mixin class for caching functionality."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

    def save_data(self, data: Any, filename: str):
        """Save data to cache."""
        if self.cache_dir:
            cache_path = self.cache_dir / filename
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

    def load_data(self, filename: str) -> Optional[Any]:
        """Load data from cache."""
        if self.cache_dir:
            cache_path = self.cache_dir / filename
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
        return None


class SearchEngine(ABC):
    """Abstract base class for search engines."""

    @abstractmethod
    def create_index(self) -> None:
        """Create search index."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Perform search operation."""
        pass


class TfidfSearch(SearchEngine, CacheMixin):
    """TF-IDF based search implementation."""

    def __init__(self, documents: List[str], cache_dir: Optional[str] = None, **tfidf_kwargs):
        super().__init__()
        CacheMixin.__init__(self, cache_dir)
        self.documents = documents
        check_invalid_input_parameters(TfidfVectorizer.__init__, tfidf_kwargs)
        self.vectorizer = TfidfVectorizer(**tfidf_kwargs)
        self.matrix = None

    def create_index(self) -> None:
        cached_data = self.load_data("tfidf_index.pkl")
        if cached_data is not None:
            self.matrix, self.vectorizer = cached_data
        else:
            self.matrix = self.vectorizer.fit_transform(self.documents)
            self.save_data((self.matrix, self.vectorizer), "tfidf_index.pkl")

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if self.matrix is None:
            raise RuntimeError("Index not created. Call create_index() first.")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [SearchResult(document=self.documents[idx], score=float(similarities[idx])) for idx in top_indices]


class HashingVectorizerSearch(SearchEngine, CacheMixin):
    """Hashing Vectorizer based search implementation."""

    def __init__(self, documents: List[str], cache_dir: Optional[str] = None, **hashing_kwargs):
        super().__init__()
        CacheMixin.__init__(self, cache_dir)
        self.documents = documents
        check_invalid_input_parameters(HashingVectorizer.__init__, hashing_kwargs)
        self.vectorizer = HashingVectorizer(**hashing_kwargs)
        self.matrix = None

    def create_index(self) -> None:
        cached_data = self.load_data("hashing_index.pkl")
        if cached_data is not None:
            self.matrix, self.vectorizer = cached_data
        else:
            self.matrix = self.vectorizer.transform(self.documents)
            self.save_data((self.matrix, self.vectorizer), "hashing_index.pkl")

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if self.matrix is None:
            raise RuntimeError("Index not created. Call create_index() first.")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [SearchResult(document=self.documents[idx], score=float(similarities[idx])) for idx in top_indices]


class SentenceTransformerSearch(SearchEngine, CacheMixin):
    """Sentence Transformer based search implementation."""

    def __init__(
        self,
        documents: List[str],
        model_name: str = "paraphrase-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        CacheMixin.__init__(self, cache_dir)
        from sentence_transformers import SentenceTransformer

        self.documents = documents
        self.model_name = model_name
        self.device = device or get_best_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embeddings = None

    def create_index(self) -> None:
        self.embeddings = self.load_data("transformer_embeddings.pkl")
        if self.embeddings is None:
            embeddings_list = []
            for text in tqdm(self.documents, desc="Creating embeddings"):
                if not text or not isinstance(text, str):
                    continue
                try:
                    embedding = self.model.encode(text, convert_to_tensor=True)
                    embeddings_list.append(embedding)
                except Exception as e:
                    logger.error(f"Error encoding text: {str(e)}")
                    continue

            if not embeddings_list:
                raise ValueError("No valid embeddings created")

            self.embeddings = torch.stack(embeddings_list)
            self.save_data(self.embeddings, "transformer_embeddings.pkl")

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Index not created. Call create_index() first.")

        if len(self.embeddings) == 0:
            return []

        try:
            # Encode query
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            query_embedding = query_embedding.to(self.embeddings.device)

            # Reshape embeddings
            query_embedding = query_embedding.view(1, -1)
            embeddings = self.embeddings.view(len(self.embeddings), -1)

            # Calculate similarities
            similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings)

            # Get top k results
            k = min(top_k, len(self.documents))
            top_values, top_indices = torch.topk(similarities, k)

            # Convert to numpy for easier handling
            top_values = top_values.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

            # Create results
            results = [
                SearchResult(document=self.documents[idx], score=float(score))
                for idx, score in zip(top_indices, top_values)
            ]

            return results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []


class LibraryInfoExtractor:
    """Extracts documentation from Python libraries."""

    def __init__(self, library_name: str):
        """
        Initialize the extractor.

        Args:
            library_name: Name of the Python library to extract info from
        """
        self.library_name = library_name
        try:
            self.library = importlib.import_module(library_name)
        except ImportError as e:
            raise ImportError(f"Could not import library {library_name}: {str(e)}")

    def extract_library_info(self) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        Extract documentation from the library.

        Yields:
            Tuples of (full_name, info_dict) where info_dict contains documentation
            and object reference
        """

        def explore_module(module, prefix="") -> Generator[Tuple[str, Dict[str, Any]], None, None]:
            if not hasattr(module, "__path__"):
                return

            for _, name, is_pkg in pkgutil.iter_modules(module.__path__):
                full_name = f"{prefix}.{name}" if prefix else name
                if "test" in name.lower():
                    continue

                try:
                    sub_module = importlib.import_module(f"{module.__name__}.{name}")
                    yield from self._extract_info_from_module(sub_module, full_name)

                    if is_pkg:
                        yield from explore_module(sub_module, full_name)
                except Exception as e:
                    logger.error(f"Error exploring {full_name}: {str(e)}")
                    continue

        try:
            yield from explore_module(self.library)
        except Exception as e:
            logger.error(f"Error exploring {self.library_name}: {str(e)}")

    def extract_unique_library_info(self) -> Dict[str, str]:
        """
        Extract unique documentation from the library.

        Returns:
            Dictionary, where key is the full object name and value is the documentation
        """
        # first use the documentations as keys to get unique docs
        unique_docs = {}
        for full_name, doc_info in self.extract_library_info():
            doc = doc_info["doc"]
            if doc not in unique_docs:
                unique_docs[doc] = full_name

        # reverse the dictionary so key will be full name and value will be doc
        return {value: key for key, value in unique_docs.items()}

    def _extract_info_from_module(
        self, module: Any, prefix: str = ""
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        Extract documentation from a specific module.

        Args:
            module: Module to extract info from
            prefix: Prefix for the full name

        Yields:
            Tuples of (full_name, info_dict) containing documentation and object reference
        """
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
                doc = inspect.getdoc(obj)
                if doc:
                    full_name = f"{prefix}.{name}" if prefix else name
                    yield full_name, {"doc": doc, "obj": obj}


class EnsembleSearchEngine:
    """Ensemble search engine combining multiple search methods."""

    def __init__(
        self,
        documents: List[str],
        methods_weights: Dict[SearchMethod, float],
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the ensemble search engine.

        Args:
            documents: List of documents to search through
            methods_weights: Dictionary mapping search methods to their weights
            cache_dir: Directory for caching search indices
        """
        self.documents = DocumentProcessor.process_documents(documents)
        self.methods_weights = methods_weights
        self.cache_dir = cache_dir
        self.engines: Dict[SearchMethod, SearchEngine] = {}
        self._initialize_engines()

    def _initialize_engines(self) -> None:
        """Initialize search engines based on specified methods and weights."""
        for method, weight in self.methods_weights.items():
            if weight <= 0 or weight > 1:
                continue

            if method == SearchMethod.TFIDF:
                engine = TfidfSearch(self.documents, cache_dir=self.cache_dir)
            elif method == SearchMethod.SENTENCE_TRANSFORMER:
                engine = SentenceTransformerSearch(self.documents, cache_dir=self.cache_dir)
            elif method == SearchMethod.HASHING:
                engine = HashingVectorizerSearch(self.documents, cache_dir=self.cache_dir)
            else:
                raise ValueError(f"Unknown search method: {method}")

            self.engines[method] = engine
            engine.create_index()

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Perform ensemble search across all initialized engines.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            DataFrame containing combined and ranked search results
        """
        all_results = []

        for method, engine in self.engines.items():
            weight = self.methods_weights[method]
            results = engine.search(query, top_k=top_k)

            for result in results:
                result.method = method.value
                result.weighted_score = result.score * weight
                all_results.append(result)

        if not all_results:
            return pd.DataFrame()

        # Convert results to DataFrame and aggregate scores
        df = pd.DataFrame([vars(r) for r in all_results])
        df["aggregated_score"] = df.groupby("document")["weighted_score"].transform("sum")

        # Return top-k unique results
        return (
            df.sort_values("aggregated_score", ascending=False)
            .drop_duplicates("document")
            .head(top_k)
            .reset_index(drop=True)
        )


def main():
    """Example usage of the ensemble search engine."""
    # Extract pandas documentation
    extractor = LibraryInfoExtractor("pandas")

    # Use a dictionary to maintain unique docs based on content
    unique_docs = extractor.extract_unique_library_info()

    # Convert back to list of dictionaries
    documents = list(unique_docs.values())

    # Configure search methods and weights
    methods_weights = {
        SearchMethod.TFIDF: 1.0,
        SearchMethod.SENTENCE_TRANSFORMER: 0.8,
        SearchMethod.HASHING: 0.6,
    }

    # Initialize and use ensemble search
    engine = EnsembleSearchEngine(documents=documents, methods_weights=methods_weights, cache_dir=None)

    # Example searches
    queries = [
        "How do I merge 2 dataframes?",
        "How to handle missing values?",
        "How to group data by column?",
    ]

    for query in queries:
        print(f"\nSearch Query: {query}")
        results = engine.search(query, top_k=10)
        print(results[["document", "method", "aggregated_score"]])


if __name__ == "__main__":
    main()
