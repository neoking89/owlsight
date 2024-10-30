"""
Include RAG (Retrieval-Augmented Generation) search functionality for specific Python libraries
(search_python_libs, LibraryInfoExtractor, PythonDocumentationProcessor).

As well as for broader use (search_documents, HashingVectorizerSearch, TfidfSearch, SentenceTransformerSearch, EnsembleSearchEngine).
"""

import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Literal, Union
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

SENTENCETRANSFORMER_DEFAULT_MODEL = "Alibaba-NLP/gte-base-en-v1.5"


def search_python_libs(
    library: str, query: str, top_k: int = 5, cache_dir: Optional[str] = None, return_context: bool = True
) -> Union[pd.DataFrame, str]:
    """
    Get search results for Python library documentation with optional formatted context.

    Parameters:
    -----------
    library : str
        Name of the Python library to search
    query : str
        Search query string
    top_k : int, default 5
        Number of top results to return
    cache_dir : Optional[str], default None
        Directory for caching search results
    return_context : bool, default True
        If True, returns formatted context string instead of DataFrame

    Returns:
    --------
    Union[pd.DataFrame, str]
        If return_context is True, returns formatted context string
        Otherwise returns DataFrame with search results
    """
    documents = PythonDocumentationProcessor.get_documents(library, cache_dir=cache_dir)

    engine = EnsembleSearchEngine(
        documents=documents,
        methods_weights={SearchMethod.TFIDF: 1.0},
        cache_dir=cache_dir,
        cache_dir_suffix=library,
    )

    results = engine.search(query, top_k=top_k)
    results["document_name"] = results["document_name"].apply(lambda x: f"{library}.{x}")

    if return_context:
        return engine.generate_context(results)

    return results


def search_documents(
    query: str,
    documents: Dict[str, str],
    top_k: int = 20,
    tfidf_weight: float = 0.3,
    sentence_transformer_weight: float = 0.7,
    sentence_transformer_model: str = SENTENCETRANSFORMER_DEFAULT_MODEL,
    cache_dir: str = ".rag_cache",
    cache_dir_suffix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Search documents using an ensemble of TFIDF and Sentence Transformer methods.

    Parameters
    ----------
    query : str
        The search query
    documents : Dict[str, str]
        A dictionary with object names as keys and documentation as values
    top_k : int, default 20
        Number of top results to return
    tfidf_weight : float, default 0.3
        Weight for the TFIDF search method
    sentence_transformer_weight : float, default 0.7
        Weight for the Sentence Transformer search method
    sentence_transformer_model : str, default "Alibaba-NLP/gte-base-en-v1.5"
        Sentence Transformer model to use
    cache_dir : str, default ".rag_cache"
        Directory for caching the embeddings and documentation
    cache_dir_suffix : str, default None
        Suffix to add to the cache directory

    Returns
    -------
    pd.DataFrame
        DataFrame containing the search results with columns:
        - document info: Information about a given document, like title, name, etc.
        - document: Documentation text
        - method: Search method used
        - score: Raw similarity score
        - weighted_score: Score weighted by method
        - aggregated_score: Combined score across methods
    """
    # Configure search methods weights
    methods_weights = {
        SearchMethod.TFIDF: tfidf_weight,
        SearchMethod.SENTENCE_TRANSFORMER: sentence_transformer_weight,
    }

    # Initialize ensemble search engine
    engine = EnsembleSearchEngine(
        documents=documents,
        methods_weights=methods_weights,
        cache_dir=cache_dir,
        cache_dir_suffix=cache_dir_suffix,
        init_arguments={
            SearchMethod.SENTENCE_TRANSFORMER: {
                "pooling_strategy": "mean",
                "model_name": sentence_transformer_model,
            }
        },
    )

    # Perform search
    results = engine.search(query, top_k=top_k)

    return results


class SearchMethod(str, Enum):
    """Supported search methods."""

    TFIDF = "tfidf"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    HASHING = "hashing"


class SearchResult(BaseModel):
    """Model to store search results with type validation."""

    document: str
    document_name: str
    score: float
    method: Optional[str] = None
    weighted_score: Optional[float] = None
    aggregated_score: Optional[float] = None


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


class LibraryInfoExtractor(CacheMixin):
    """Extracts documentation from Python libraries."""

    def __init__(self, library_name: str, cache_dir: Optional[str] = None, cache_dir_suffix: Optional[str] = None):
        """Initialize the extractor."""
        super().__init__(cache_dir, cache_dir_suffix)
        self.library_name = library_name
        try:
            self.library = importlib.import_module(library_name)
        except ImportError as e:
            raise ImportError(f"Could not import library {library_name}: {str(e)}")

    def extract_library_info(self) -> Dict[str, str]:
        """Extract documentation from the library."""
        if self.cache_dir:
            cached_data = self.load_data()
            if cached_data is not None:
                return cached_data

        unique_docs = {}
        # add documentation as key to keep it unique
        for full_name, doc_info in self._extract_library_info_as_generator():
            unique_docs[doc_info["doc"]] = full_name

        # afterwards, reverse the key-value pairs to have the object name as key
        unique_docs = {name: doc for doc, name in unique_docs.items() if doc}

        if self.cache_dir:
            self.save_data(unique_docs)

        return unique_docs

    @staticmethod
    def import_from_string(path: str) -> Any:
        """
        Import a class or function from a string path.

        Parameters:
        ----------
        path: str
            The path to the class or function to import.
            Example: "pandas.DataFrame"
        """
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _extract_library_info_as_generator(self) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Extract documentation from the library."""

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

    def _extract_info_from_module(
        self, module: Any, prefix: str = ""
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Extract documentation from a specific module."""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
                doc = inspect.getdoc(obj)
                if doc:
                    full_name = f"{prefix}.{name}" if prefix else name
                    yield full_name, {"doc": doc, "obj": obj}


class PythonDocumentationProcessor:
    """
    Handles document preprocessing and validation specific to Python libraries.
    """

    @staticmethod
    def process_documents(documents: Dict[str, str]) -> Dict[str, str]:
        """Process and validate input documents."""
        processed_docs = {}

        for obj_name, doc in documents.items():
            if isinstance(doc, str):
                processed_docs[obj_name] = doc
            elif hasattr(doc, "__doc__") and doc.__doc__:
                processed_docs[obj_name] = doc.__doc__

        if not processed_docs:
            raise ValueError("No valid documents found after processing")
        return processed_docs

    @staticmethod
    def get_documents(lib: str, cache_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Get documentation for a Python library.
        Involves all the necessary steps to extract and process documentation.

        Parameters:
        ----------
        lib: str
            The name of the library to extract documentation from.
        cache_dir: Optional[str]
            The cache directory to store the extracted documentation.

        Returns:
        -------
        Dict[str, str]
            A dictionary with object names as keys and documentation as values.
        """
        extractor = LibraryInfoExtractor(lib, cache_dir=cache_dir, cache_dir_suffix=lib)
        docs_with_names = extractor.extract_library_info()
        docs_with_names = PythonDocumentationProcessor.process_documents(docs_with_names)
        return docs_with_names


class SearchEngine(ABC):
    """Abstract base class for search engines."""

    @property
    def cls_name(self) -> str:
        """Get class name."""
        return self.__class__.__name__

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

    def __init__(
        self,
        documents: Dict[str, str],
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
        **tfidf_kwargs,
    ):
        super().__init__()
        check_invalid_input_parameters(TfidfVectorizer.__init__, tfidf_kwargs)
        if cache_dir_suffix:
            cache_dir_suffix = f"{self.cls_name}__{cache_dir_suffix}"

        CacheMixin.__init__(self, cache_dir, cache_dir_suffix)
        self.documents = documents
        self.doc_list = list(documents.values())
        self.obj_names = list(documents.keys())
        self.vectorizer = TfidfVectorizer(**tfidf_kwargs)
        self.matrix = None

    def create_index(self) -> None:
        cached_data = self.load_data()
        if cached_data is not None:
            self.matrix, self.vectorizer = cached_data
        else:
            self.matrix = self.vectorizer.fit_transform(self.doc_list)
            self.save_data((self.matrix, self.vectorizer))

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if self.matrix is None:
            raise RuntimeError("Index not created. Call create_index() first.")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            SearchResult(document=self.doc_list[idx], document_name=self.obj_names[idx], score=float(similarities[idx]))
            for idx in top_indices
        ]


class HashingVectorizerSearch(SearchEngine, CacheMixin):
    """Hashing Vectorizer based search implementation."""

    def __init__(
        self,
        documents: Dict[str, str],
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
        **hashing_kwargs,
    ):
        """Initialize the HashingVectorizer search engine."""
        super().__init__()
        check_invalid_input_parameters(HashingVectorizer.__init__, hashing_kwargs)
        if cache_dir_suffix:
            cache_dir_suffix = f"{self.cls_name}__{cache_dir_suffix}"

        CacheMixin.__init__(self, cache_dir, cache_dir_suffix)
        self.documents = documents
        self.doc_list = list(documents.values())
        self.obj_names = list(documents.keys())
        self.vectorizer = HashingVectorizer(**hashing_kwargs)
        self.matrix = None

    def create_index(self) -> None:
        cached_data = self.load_data()
        if cached_data is not None:
            self.matrix, self.vectorizer = cached_data
        else:
            self.matrix = self.vectorizer.transform(self.doc_list)
            self.save_data((self.matrix, self.vectorizer))

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if self.matrix is None:
            raise RuntimeError("Index not created. Call create_index() first.")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            SearchResult(document=self.doc_list[idx], document_name=self.obj_names[idx], score=float(similarities[idx]))
            for idx in top_indices
        ]


class SentenceTransformerSearch(SearchEngine, CacheMixin):
    """Sentence Transformer based search implementation."""

    def __init__(
        self,
        documents: Dict[str, str],
        model_name: str = SENTENCETRANSFORMER_DEFAULT_MODEL,
        pooling_strategy: Literal["mean", "max", None] = "mean",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
    ):
        """Initialize the Sentence Transformer search engine."""
        self._check_pooling_strategy(pooling_strategy)
        if cache_dir_suffix:
            cache_dir_suffix = (
                f"{self.cls_name}__{cache_dir_suffix}____{pooling_strategy}__{model_name.replace('/', '_')}"
            )

        super().__init__()
        CacheMixin.__init__(self, cache_dir, cache_dir_suffix)
        from sentence_transformers import SentenceTransformer

        self.documents = documents
        self.doc_list = list(documents.values())
        self.obj_names = list(documents.keys())
        self.model_name = model_name
        self.device = device or get_best_device()
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        self.embeddings = None
        self._pooling_strategy = pooling_strategy

    def create_index(self) -> None:
        self.embeddings = self.load_data()
        if self.embeddings is None:
            embeddings_list = []
            for text in tqdm(self.doc_list, desc="Creating embeddings"):
                if not text or not isinstance(text, str):
                    continue
                try:
                    if self._pooling_strategy:
                        text = self.split_and_clean_text(text)
                    embedding = self.model.encode(text, convert_to_tensor=True)
                    # apply mean or max pooling to get one fixed-size embedding
                    if self._pooling_strategy == "mean":
                        embedding = torch.mean(embedding, dim=0)
                    elif self._pooling_strategy == "max":
                        embedding = torch.max(embedding, dim=0)[0]
                    embeddings_list.append(embedding)
                except Exception as e:
                    logger.error(f"Error encoding text: {str(e)}")
                    continue

            if not embeddings_list:
                raise ValueError("No valid embeddings created")

            self.embeddings = torch.stack(embeddings_list)
            self.save_data(self.embeddings)

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Index not created. Call create_index() first.")

        if len(self.embeddings) == 0:
            return []

        try:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            query_embedding = query_embedding.to(self.embeddings.device)
            query_embedding = query_embedding.view(1, -1)
            embeddings = self.embeddings.view(len(self.embeddings), -1)
            similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
            k = min(top_k, len(self.doc_list))
            top_values, top_indices = torch.topk(similarities, k)
            top_values = top_values.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

            return [
                SearchResult(document=self.doc_list[idx], document_name=self.obj_names[idx], score=float(score))
                for idx, score in zip(top_indices, top_values)
            ]

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

    @staticmethod
    def split_and_clean_text(text: str) -> list:
        """Split a longer text into sentences and clean them."""
        cleaned_text = text.replace("\n", " ")
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", cleaned_text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _check_pooling_strategy(self, pooling_strategy: Optional[str]) -> None:
        pooling_choices = [None, "mean", "max"]
        if pooling_strategy not in pooling_choices:
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}. Pooling choices: {pooling_choices}")
        self._pooling_strategy = pooling_strategy


class EnsembleSearchEngine:
    """Ensemble search engine combining multiple search methods."""

    def __init__(
        self,
        documents: Dict[str, str],
        methods_weights: Dict[SearchMethod, float],
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
        init_arguments: Optional[Dict[str, Dict]] = None,
    ):
        """
        Initialize the ensemble search engine.

        Parameters:
        -----------
        documents : Dict[str, str]
            Dictionary containing document names and content
        methods_weights : Dict[SearchMethod, float]
            Dictionary containing search methods and their corresponding weights
        cache_dir : Optional[str], default None
            Directory for caching search results
        cache_dir_suffix : Optional[str], default None
            Suffix to append to cache directory. Required if cache_dir is specified
        init_arguments : Optional[Dict[str, Dict]], default None
            Dictionary containing initialization arguments for each search method
            Example: {SearchMethod.TFIDF: {"ngram_range": (1, 2)}}
        """
        self.documents = documents
        self.methods_weights = methods_weights
        self.cache_dir = cache_dir
        self.cache_dir_suffix = cache_dir_suffix
        self.engines: Dict[SearchMethod, SearchEngine] = {}
        self.engine_init_arguments = init_arguments or {}
        self._initialize_engines()

    def _initialize_engines(self) -> None:
        """Initialize search engines based on specified methods and weights."""
        for method, weight in self.methods_weights.items():
            if weight <= 0:
                continue

            kwargs = {
                "documents": self.documents,
                "cache_dir": self.cache_dir,
                "cache_dir_suffix": self.cache_dir_suffix or "",
            }

            if method == SearchMethod.TFIDF:
                engine = TfidfSearch(**kwargs | self.engine_init_arguments.get(SearchMethod.TFIDF, {}))
            elif method == SearchMethod.SENTENCE_TRANSFORMER:
                engine = SentenceTransformerSearch(
                    **kwargs | self.engine_init_arguments.get(SearchMethod.SENTENCE_TRANSFORMER, {})
                )
            elif method == SearchMethod.HASHING:
                engine = HashingVectorizerSearch(**kwargs | self.engine_init_arguments.get(SearchMethod.HASHING, {}))
            else:
                raise ValueError(f"Unknown search method: {method}")

            self.engines[method] = engine
            engine.create_index()

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Perform ensemble search across all initialized engines."""
        index_name = "document_name"
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
        df["aggregated_score"] = df.groupby(index_name)["weighted_score"].transform("sum")

        # Return top-k unique results, using document_name as index
        return (
            df.sort_values("aggregated_score", ascending=False)
            .drop_duplicates(index_name)
            .head(top_k)
            .set_index(index_name)
            .reset_index()
        )

    def generate_context(self, results: pd.DataFrame) -> str:
        """
        Generate formatted context from search results.

        Parameters:
        -----------
        results : pd.DataFrame
            Search results DataFrame containing document names and content

        Returns:
        --------
        str
            Formatted context string
        """
        context_parts = []

        for _, row in results.iterrows():
            # Get object from full path
            try:
                obj = LibraryInfoExtractor.import_from_string(row["document_name"])
                signature = _get_signature(obj)
            except Exception as e:
                logger.warning(f"Error getting object info: {str(e)}")
                signature = "(Unable to retrieve signature)"

            # Format header with name, signature, and score
            header = f"{row['document_name']}{signature}"
            # score_info = f"(Relevance Score: {row['score']:.3f})"

            # Build context entry
            entry = ["=" * 80, header, "-" * 40, "Documentation:", row["document"].strip(), "\n"]

            context_parts.append("\n".join(entry))

        # Combine all entries
        return "\n".join(context_parts)


def _get_signature(obj: Any) -> str:
    """
    Get the signature of a callable object.

    Parameters:
    -----------
    obj : Any
        Object to get signature for

    Returns:
    --------
    str
        String representation of the object's signature
    """
    try:
        if callable(obj):
            return str(inspect.signature(obj))
        return ""
    except (ValueError, TypeError):
        return "(Unable to retrieve signature)"


# if __name__ == "__main__":
#     import time

#     start = time.time()
#     results = get_python_lib_results(
#         library="pandas", query="create DataFrame from list", top_k=5, cache_dir=".rag_cache"
#     )
#     print(results)
#     print(f"Time taken: {time.time() - start:.2f} seconds")
