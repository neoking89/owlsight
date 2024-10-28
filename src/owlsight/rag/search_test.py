import sys
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Literal
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

LoggerManager.configure_logger(log_path="logs")
logger = LoggerManager.get_logger(__name__)


class SearchMethod(str, Enum):
    """Supported search methods."""

    TFIDF = "tfidf"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    HASHING = "hashing"


class SearchResult(BaseModel):
    """Model to store search results with type validation."""

    document: str
    object_name: str
    score: float
    method: Optional[str] = None
    weighted_score: Optional[float] = None
    aggregated_score: Optional[float] = None


class PythonDocumentationProcessor:
    """Handles document preprocessing and validation."""

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
            SearchResult(document=self.doc_list[idx], object_name=self.obj_names[idx], score=float(similarities[idx]))
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
        """
        Initialize the HashingVectorizer search engine.

        Parameters:
        ----------
        documents: Dict[str, str]
            A dictionary with name/documentinfo as keys and documents as values.
        cache_dir: Optional[str]
            The cache directory to store the embeddings.


        """

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
            SearchResult(document=self.doc_list[idx], object_name=self.obj_names[idx], score=float(similarities[idx]))
            for idx in top_indices
        ]


class SentenceTransformerSearch(SearchEngine, CacheMixin):
    """Sentence Transformer based search implementation."""

    def __init__(
        self,
        documents: Dict[str, str],
        model_name: str = "Alibaba-NLP/gte-base-en-v1.5",
        pooling_strategy: Literal["mean", "max", None] = "mean",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
    ):
        """
        Initialize the Sentence Transformer search engine

        Parameters:
        ----------
        documents: Dict[str, str]
            A dictionary with name/documentinfo as keys and documents as values.
        model_name: str
            The name of the Sentence Transformer model to use.
        pooling_strategy: Optional[Literal["mean", "max"]]
            The pooling strategy to apply to the document embeddings.
            Options: "mean", "max" or None.
            If None, the document embeddings will be used as is.
            If a pooling strategy is provided, the document will be split into sentences and the pooling strategy will be applied.
        device: Optional[str]
            The device to use for the Sentence Transformer model.
        cache_dir: Optional[str]
            The cache directory to store the embeddings.
        cache_dir_suffix: Optional[str]
            The cache directory suffix to store the embeddings.
            Must be provided if cache_dir is specified.
        """
        self._check_pooling_strategy(pooling_strategy)
        if cache_dir_suffix:
            cache_dir_suffix = (
                f"{self.cls_name}__{cache_dir_suffix}____{pooling_strategy}__{model_name.replace('/', '_')}"
            )

        _document_warning = f"""
When using {self.cls_name}, ensure that documents are split into sentences.
If not, use a pooling strategy, which will split a document into sentences and apply pooling to get one fixed-size embedding.
Also, always check the documentation of the model to check if there is additional preprocessing required.
Current pooling strategy: '{pooling_strategy}'
        """.strip()
        logger.warning(_document_warning)

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
                SearchResult(document=self.doc_list[idx], object_name=self.obj_names[idx], score=float(score))
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
            raise ValueError(f"Invalid pooling strategy: {pooling_strategy}. Pooling choices:{pooling_choices}")
        self._pooling_strategy = pooling_strategy


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


class EnsembleSearchEngine:
    """Ensemble search engine combining multiple search methods."""

    def __init__(
        self,
        documents: Dict[str, str],
        methods_weights: Dict[SearchMethod, float],
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
        init_arguments: Optional[Dict[str, SearchMethod]] = None,
    ):
        """Initialize the ensemble search engine."""
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
        df["aggregated_score"] = df.groupby("object_name")["weighted_score"].transform("sum")

        # Return top-k unique results, using object_name as index
        return (
            df.sort_values("aggregated_score", ascending=False)
            .drop_duplicates("object_name")
            .head(top_k)
            .set_index("object_name")
            .reset_index()
        )


def search_documents(
    query: str,
    documents: Dict[str, str],
    top_k: int = 20,
    tfidf_weight: float = 0.3,
    sentence_transformer_weight: float = 0.7,
    sentence_transformer_model: str = "multilingual-e5-large-instruct",
    reranker_model: Optional[str] = None,
    rerank_top_n_sentences: int = 10,
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
    reranker_model : str, default None
        Path to the reranker model
    rerank_top_n_sentences : int, default 10
        Number of top sentences to consider for reranking. The mean of these scores is used to rank the documents.
        The idea is that length of the document should not affect the ranking.
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
    if reranker_model:
        results = rerank_results(query, reranker_model, rerank_top_n_sentences, results)

    return results


def rerank_results(query, reranker_model, rerank_top_n_sentences, results):
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(reranker_model, device=get_best_device())
    texts = results["document"].tolist()
    split_documents = [SentenceTransformerSearch.split_and_clean_text(text) for text in texts]
    # documents are now split into sentences. Now we need to make query and sentence pairs per document
    results["rerank_score"] = 0.0

    for idx, document in enumerate(tqdm(split_documents, desc="Reranking documents")):
        query_sentence_pairs = [[query, sentence] for sentence in document]
        # encode the query and sentence pairs
        pair_scores = reranker.predict(query_sentence_pairs)
        # take the top n sentence scores so that document size matters less.
        # Problem is document can be very relevant, but can have lots of low-scoring sentences
        top_n_indices = np.argsort(pair_scores)[-rerank_top_n_sentences:][::-1]
        rank_score = np.mean(pair_scores[top_n_indices])
        results.loc[idx, "rerank_score"] = rank_score

    return results.sort_values("rerank_score", ascending=False)


def main():
    """Example usage of the ensemble search engine."""
    # Extract pandas documentation
    cache_dir = ".rag_cache"

    lib = "pandas"
    docs_with_names = PythonDocumentationProcessor.get_documents(lib, cache_dir=cache_dir)

    # Example searches
    queries = [
        "read csv file",
        "plot data",
        "group by column",
        "merge dataframes",
        "create new column",
        "drop na values",
        "pivot table",
        "time series data",
    ]

    for query in queries:
        logger.info(f"\nSearch Query: {query}")
        results = search_documents(query, docs_with_names, top_k=10, cache_dir=cache_dir, cache_dir_suffix=lib)
        print(results)


if __name__ == "__main__":
    main()
