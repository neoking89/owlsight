import re
import os
from typing import Dict, List, Optional, Literal, Any, Union, Tuple
from abc import ABC, abstractmethod
import traceback

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from owlsight.rag.custom_classes import CacheMixin, SearchMethod, SearchResult
from owlsight.rag.helper_functions import _get_signature
from owlsight.rag.constants import SENTENCETRANSFORMER_DEFAULT_MODEL
from owlsight.utils.deep_learning import get_best_device
from owlsight.utils.helper_functions import check_invalid_input_parameters
from owlsight.utils.logger import logger


class DocumentSearcher:
    """Document search engine using an ensemble of TFIDF and Sentence Transformer methods.

    This class provides document search capability by combining traditional TF-IDF
    with modern neural embeddings. The idea behind this is two-fold:
    - TFIDF can capture relevant words an embedding model was not trained on.
    - Embeddings can capture context better than TFIDF.

    Order in __init__is like so:
    [splitting in chunks (optional)]
    [TF-IDF]
    [Sentence Transformer: create embeddings and cache as .pkl files]

    And then use the `search` method to combine the results:
    [Combine TF-IDF and Sentence Transformer results]

    Parameters
    ----------
    documents : Dict[str, str]
        Dictionary mapping document IDs to their content
    sentence_transformer_model : str, default='Alibaba-NLP/gte-base-en-v1.5'
        Name or path of the Sentence Transformer model
    sentence_transformer_batch_size : int, default=64
        Batch size for computing embeddings
    split_documents_n_sentences : int, optional
        Number of sentences to split each document into.
        If None, no splitting is done per document
        If not None, splitting is done per document.
        Also, documents will be split into chunks of size `split_documents_n_sentences` so that
        the number of sentences is equal to `split_documents_n_sentences`.
    split_documents_n_overlap : int, default=1
        Only applies if `split_documents_n_sentences` is not None.
        Number of sentences to overlap between adjacent chunks.
    cache_dir : str, optional
        Directory to cache embeddings and results
    cache_dir_suffix : str, optional
        Suffix for cache directory name.
        Recommended is to use a name which is unique and descriptive to the documents.

    Notes
    -----
    - Uses both TF-IDF and neural embeddings for robust search
    - Has caching capabilities in pickled files
    - Supports batch processing for efficient embedding computation

    Examples
    --------
    >>> docs = {
    ...     "doc1": "Python is a programming language",
    ...     "doc2": "Machine learning is fascinating"
    ... }
    >>> searcher = DocumentSearcher(docs, cache_dir="document_cache", cache_dir_suffix="programming")
    >>> results = searcher.search("python programming", top_k=3)
    """

    def __init__(
        self,
        documents: Dict[str, str],
        sentence_transformer_model: str = SENTENCETRANSFORMER_DEFAULT_MODEL,
        sentence_transformer_batch_size: int = 64,
        split_documents_n_sentences: Optional[int] = None,
        split_documents_n_overlap: int = 1,
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
    ) -> None:
        if not documents and not cache_dir:
            raise ValueError("documents must not be empty if cache_dir is not provided")

        self.sentence_transformer_model = sentence_transformer_model
        self.sentence_transformer_batch_size = sentence_transformer_batch_size

        self.cache_dir = cache_dir
        self.documents = documents
        self._handle_cache_and_documents(cache_dir_suffix, split_documents_n_sentences, split_documents_n_overlap)

        self.split_documents_n_sentences = split_documents_n_sentences
        self.split_documents_n_overlap = split_documents_n_overlap

        engine_init_arguments = {
            SearchMethod.SENTENCE_TRANSFORMER: {
                "pooling_strategy": "mean",
                "model_name": self.sentence_transformer_model,
                "batch_size": self.sentence_transformer_batch_size,
            }
        }
        self.engine = EnsembleSearchEngine(
            documents=self.documents,
            search_methods=[SearchMethod.TFIDF, SearchMethod.SENTENCE_TRANSFORMER],
            cache_dir=self.cache_dir,
            cache_dir_suffix=self.cache_dir_suffix,
            init_arguments=engine_init_arguments,
        )

    @classmethod
    def from_cache(cls, cache_dir: str, cache_dir_suffix: str, **init_kwargs) -> "DocumentSearcher":
        """Load a DocumentSearcher instance from cache."""
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

        pkl_files = [f[:-4] for f in os.listdir(cache_dir) if f.endswith(".pkl")]
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in cache directory: {cache_dir}")

        if not any(map(lambda file: cache_dir_suffix == file.split("__")[1], pkl_files)):
            raise FileNotFoundError(
                f"No cache files found in dir {cache_dir} with cache_dir_suffix: {cache_dir_suffix}"
            )

        init_cls = cls(
            documents={},
            cache_dir=cache_dir,
            cache_dir_suffix=cache_dir_suffix,
            **init_kwargs,
        )

        return init_cls

    @staticmethod
    def split_documents(
        documents: Dict[str, str],
        n_sentences: int = 3,
        n_overlap: int = 0,
    ) -> Dict[str, str]:
        """Split documents into chunks of n sentences with overlap.

        Parameters
        ----------
        documents : Dict[str, str]
            Dictionary of documents to split, where keys are document names and values are document texts.
        n_sentences : int, default=3
            Number of sentences per chunk
        n_overlap : int, default=1
            Number of sentences to overlap between chunks

        Returns
        -------
        Dict[str, str]
            Dictionary with new document names as keys and sentence chunks as values.
            New document names follow the pattern: [document_name]__split[n]
        """
        if n_overlap >= n_sentences:
            raise ValueError("n_overlap must be less than n_sentences")

        split_docs = {}
        for doc_name, doc_text in documents.items():
            # Handle empty documents
            if not doc_text.strip():
                split_docs[f"{doc_name}__split0"] = ""
                continue

            # Split document into sentences
            sentences = SentenceTransformerSearchEngine.split_and_clean_text(doc_text)

            # Handle documents with no proper sentences
            if not sentences:
                split_docs[f"{doc_name}__split0"] = doc_text
                continue

            # If document is shorter than chunk size, keep it as one chunk
            if len(sentences) <= n_sentences:
                split_docs[f"{doc_name}__split0"] = " ".join(sentences)
                continue

            # Create chunks with overlap
            chunk_idx = 0
            start_idx = 0

            while start_idx < len(sentences):
                # Get chunk of n_sentences (or remaining sentences)
                end_idx = min(start_idx + n_sentences, len(sentences))
                chunk = sentences[start_idx:end_idx]

                # Join sentences into a single string
                chunk_text = " ".join(chunk)

                # Create new document name with chunk index
                new_doc_name = f"{doc_name}__split{chunk_idx}"
                split_docs[new_doc_name] = chunk_text
                chunk_idx += 1

                # Move start index forward by (n_sentences - n_overlap)
                start_idx += n_sentences - n_overlap

                # If we can't form another complete chunk, append the remaining sentences
                if len(sentences) - start_idx < n_sentences:
                    split_docs[new_doc_name] = " ".join(chunk)

        return split_docs

    def search(
        self,
        query: str,
        top_k: int = 20,
        sentence_transformer_weight: float = 0.7,
        tfidf_weight: float = 0.3,
        as_context: bool = False,
    ) -> Union[pd.DataFrame, str]:
        """
        Search documents using the configured ensemble methods.

        Parameters
        ----------
        query : str
            The search query
        top_k : int, default 20
            Number of top results to return
        tfidf_weight : float, default 0.3
            Weight for the TFIDF search method.
        sentence_transformer_weight : float, default 0.7
            Weight for the Sentence Transformer search method.
        as_context : bool, default False
            If True, format the DataFrame as string, focussing on being passed as context for LLMs.

        Returns
        -------
        Union[pd.DataFrame, str]
            DataFrame containing the search results or a formatted string if `as_context` is True.
        """
        method_weights = {
            SearchMethod.TFIDF: tfidf_weight,
            SearchMethod.SENTENCE_TRANSFORMER: sentence_transformer_weight,
        }
        results = self.engine.search(query, top_k=top_k, method_weights=method_weights)

        if as_context:
            context_parts = []
            for _, row in results.iterrows():
                header = f"{row['document_name']}"
                entry = ["=" * 80, header, "-" * 40, "Content:", row["document"].strip()]
                context_parts.append("\n".join(entry))
            return "".join(context_parts)

        return results

    def _handle_split_documents(
        self,
        documents: Dict[str, str],
        split_documents_n_sentences: Optional[int],
        split_documents_n_overlap: int,
    ) -> Dict[str, str]:
        """Handle document splitting based on instance configuration.

        Parameters
        ----------
        documents : Dict[str, str]
            Input documents to potentially split

        Returns
        -------
        Dict[str, str]
            Documents after potential splitting

        Raises
        ------
        ValueError
            If split_documents_n_overlap is greater than or equal to split_documents_n_sentences
        """
        if split_documents_n_sentences is not None:
            if split_documents_n_overlap >= split_documents_n_sentences:
                raise ValueError("split_documents_n_overlap must be less than split_documents_n_sentences")
            documents = self.split_documents(
                documents,
                n_sentences=split_documents_n_sentences,
                n_overlap=split_documents_n_overlap,
            )
        return documents

    def _create_unique_cache_dir_suffix(
        self, cache_dir_suffix: str, split_documents_n_sentences: str, split_documents_n_overlap: str
    ) -> str:
        """Create unique cache names based on instance configuration"""
        cache_dir_suffix = f"{cache_dir_suffix}__split_documents_n_sentences={split_documents_n_sentences}__split_documents_n_overlap={split_documents_n_overlap}"
        return cache_dir_suffix

    def _handle_cache_and_documents(self, cache_dir_suffix, split_documents_n_sentences, split_documents_n_overlap):
        if self.cache_dir and cache_dir_suffix:
            self.cache_dir_suffix = self._create_unique_cache_dir_suffix(
                cache_dir_suffix, split_documents_n_sentences, split_documents_n_overlap
            )
            cache_mixin = CacheMixin(
                cache_dir=self.cache_dir,
                cache_dir_suffix=f"cache_dir_suffix={self.cache_dir_suffix}",
            )
            if cache_mixin.get_full_cache_path().exists() and not self.documents:
                self.documents = cache_mixin.load_data()
            else:
                self.documents = self._handle_split_documents(
                    self.documents, split_documents_n_sentences, split_documents_n_overlap
                )
                cache_mixin.save_data(self.documents)
        else:
            self.documents = self._handle_split_documents(
                self.documents, split_documents_n_sentences, split_documents_n_overlap
            )


class SearchEngine(ABC):
    """Abstract base class for all search engine implementations.

    This class defines the interface that all search engines must implement.
    Subclasses should implement create_index() and search() methods.

    Methods
    -------
    create_index()
        Create search index from documents
    search(query: str, top_k: int = 3)
        Search documents using the query
    """

    @property
    def cls_name(self) -> str:
        """Get class name."""
        return self.__class__.__name__

    @abstractmethod
    def create_index(self) -> None:
        """Create search index from documents."""

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Search documents using the query.

        Parameters
        ----------
        query : str
            Search query text
        top_k : int, default=3
            Number of top results to return

        Returns
        -------
        List[SearchResult]
            List of search results, ordered by relevance
        """


class TFIDFSearchEngine(SearchEngine, CacheMixin):
    """Search engine using TF-IDF (Term Frequency-Inverse Document Frequency).

    This search engine uses traditional TF-IDF vectorization for keyword-based search,
    making it effective for finding documents with specific terms.

    Parameters
    ----------
    documents : Dict[str, str]
        Dictionary mapping document IDs to their content
    cache_dir : str, optional
        Directory to cache TF-IDF matrices
    cache_dir_suffix : str, optional
        Suffix for cache directory name
    **tfidf_kwargs
        Additional arguments passed to sklearn.feature_extraction.text.TfidfVectorizer

    Notes
    -----
    - Fast and memory-efficient
    - Good for exact keyword matching
    - Supports n-grams and custom tokenization
    - Caches TF-IDF matrices for better performance

    Examples
    --------
    >>> docs = {
    ...     "doc1": "Python programming basics",
    ...     "doc2": "Advanced Python concepts"
    ... }
    >>> engine = TFIDFSearchEngine(docs, ngram_range=(1, 2))
    >>> results = engine.search("python basics", top_k=1)
    """

    def __init__(
        self,
        documents: Dict[str, str],
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
        **tfidf_kwargs: Any,
    ) -> None:
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


class HashingVectorizerSearchEngine(SearchEngine, CacheMixin):
    """Search engine using Hashing Vectorizer for memory-efficient search.

    This search engine uses feature hashing for vectorization, making it memory-efficient
    and suitable for large document collections.

    Parameters
    ----------
    documents : Dict[str, str]
        Dictionary mapping document IDs to their content
    cache_dir : str, optional
        Directory to cache hash matrices
    cache_dir_suffix : str, optional
        Suffix for cache directory name
    **hashing_kwargs
        Additional arguments passed to sklearn.feature_extraction.text.HashingVectorizer

    Notes
    -----
    - Memory-efficient, suitable for large datasets
    - No inverse transform capability
    - Constant memory usage regardless of vocabulary size
    - Small chance of hash collisions

    Examples
    --------
    >>> docs = {
    ...     "doc1": "Large text document...",
    ...     "doc2": "Another large document..."
    ... }
    >>> engine = HashingVectorizerSearchEngine(
    ...     docs,
    ...     n_features=(2**16)
    ... )
    >>> results = engine.search("specific terms", top_k=1)
    """

    def __init__(
        self,
        documents: Dict[str, str],
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
        **hashing_kwargs: Any,
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


class SentenceTransformerSearchEngine(SearchEngine, CacheMixin):
    """Search engine using Sentence Transformer embeddings.

    This search engine uses neural embeddings to find semantically similar documents,
    making it effective for concept-based search rather than just keyword matching.

    Parameters
    ----------
    documents : Dict[str, str]
        Dictionary mapping document IDs to their content
    model_name : str, default='Alibaba-NLP/gte-base-en-v1.5'
        Name or path of the Sentence Transformer model
    pooling_strategy : {'mean', 'max', None}, default='mean'
        Strategy for pooling sentence embeddings:
        - 'mean': Average embeddings (better for context)
        - 'max': Maximum values (better for key concepts)
        - None: No pooling (for single-sentence documents)
    device : str, optional
        Device to run model on ('cpu', 'cuda', etc.)
    cache_dir : str, optional
        Directory to cache embeddings
    cache_dir_suffix : str, optional
        Suffix for cache directory name
    batch_size : int, default=64
        Batch size for computing embeddings

    Notes
    -----
    - Provides semantic search capability
    - Automatically handles sentence splitting and pooling
    - Supports GPU acceleration
    - Caches embeddings for better performance

    Examples
    --------
    >>> docs = {
    ...     "doc1": "Python is great for machine learning",
    ...     "doc2": "Deep learning revolutionized AI"
    ... }
    >>> engine = SentenceTransformerSearchEngine(
    ...     docs,
    ...     model_name='all-MiniLM-L6-v2',
    ...     pooling_strategy='mean'
    ... )
    >>> results = engine.search("AI and ML", top_k=1)
    """

    def __init__(
        self,
        documents: Dict[str, str],
        model_name: str = SENTENCETRANSFORMER_DEFAULT_MODEL,
        pooling_strategy: Literal["mean", "max", None] = "mean",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
        batch_size: int = 64,
    ):
        """
        Initialize the Sentence Transformer search engine.

        Parameters:
        -----------
        documents : Dict[str, str]
            Dictionary containing document names and content
        model_name : str
            Sentence Transformer model name. Can also be a path to a local model.
        pooling_strategy : Literal["mean", "max", None], default "mean"
            Pooling strategy to use for Sentence Transformer embeddings
            Use "mean" or "max" for mean or max pooling, respectively.
            This is useful when the input text has multiple sentences, but you want a single embedding which maintains the context.
            Splitting of sentences is done automatically.
            Choose None for no pooling. This is useful if each document is a single sentence.
        device : Optional[str], default None
            Device to use for Sentence Transformer model
        cache_dir : Optional[str], default None
            Directory for caching search results
        cache_dir_suffix : Optional[str], default None
            Suffix to append to cache directory. Required if cache_dir is specified
        batch_size : int, default 32
            Batch size for embedding creation
        """
        self._check_pooling_strategy(pooling_strategy)
        if cache_dir_suffix:
            cache_dir_suffix = f"{self.cls_name}__{cache_dir_suffix}____pooling={pooling_strategy}__model={model_name.replace('/', '_')}"

        super().__init__()
        CacheMixin.__init__(self, cache_dir, cache_dir_suffix)
        from sentence_transformers import SentenceTransformer

        self.documents = documents
        self.doc_list = list(documents.values())
        self.obj_names = list(documents.keys())
        self.model_name = model_name
        self.device = device or get_best_device()
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        self.batch_size = batch_size
        self.embeddings = None
        self._pooling_strategy = pooling_strategy

    def create_index(self) -> None:
        """Create search index by computing embeddings for all documents.

        This method:
        1. Splits documents into sentences
        2. Computes embeddings in batches
        3. Applies pooling if specified
        4. Caches results for future use
        """
        self.embeddings = self.load_data()
        # inmediately return if embeddings are already cached
        if self.embeddings is not None:
            return

        # Pre-filter valid texts and prepare them
        valid_texts = []

        for text in self.doc_list:
            if text and isinstance(text, str):
                if self._pooling_strategy:
                    sentences = self.split_and_clean_text(text)
                    if sentences:  # Only add if we have valid sentences
                        valid_texts.append(sentences if self._pooling_strategy else text)
                else:
                    valid_texts.append(text)

        if not valid_texts:
            raise ValueError("No valid texts found for embedding creation")

        try:
            # Batch encode all texts at once

            embeddings_list = []
            if self.cache_dir and self.cache_dir_suffix:
                logger.info("Embeddings will be cached in %s", self.get_full_cache_path())

            for i in tqdm(range(0, len(valid_texts), self.batch_size), desc="Creating embeddings"):
                batch_texts = valid_texts[i : i + self.batch_size]

                # Handle both single texts and lists of sentences
                if self._pooling_strategy:
                    # Flatten the list of sentences for batch processing
                    flat_sentences = [sent for doc in batch_texts for sent in doc]
                    if not flat_sentences:
                        continue

                    # Encode flattened sentences
                    batch_embeddings = self.model.encode(
                        flat_sentences, convert_to_tensor=True, show_progress_bar=False
                    )

                    # Reshape embeddings back to document structure
                    start_idx = 0
                    doc_embeddings = []
                    for doc in batch_texts:
                        if not doc:  # Skip empty documents
                            continue
                        doc_size = len(doc)
                        doc_embedding = batch_embeddings[start_idx : start_idx + doc_size]
                        start_idx += doc_size

                        # Apply pooling per document
                        if self._pooling_strategy == "mean":
                            doc_embedding = torch.mean(doc_embedding, dim=0)
                        else:  # max pooling
                            doc_embedding = torch.max(doc_embedding, dim=0)[0]
                        doc_embeddings.append(doc_embedding)

                    if doc_embeddings:
                        embeddings_list.append(torch.stack(doc_embeddings))
                else:
                    # Direct encoding for single texts
                    batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
                    embeddings_list.append(batch_embeddings)

            if not embeddings_list:
                raise ValueError("No embeddings were created")

            # Concatenate all batches
            self.embeddings = torch.cat(embeddings_list, dim=0)
            self.save_data(self.embeddings)

        except Exception as e:
            logger.error(f"Error in batch embedding creation: {str(e)}")
            raise ValueError("Failed to create embeddings") from e

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
            logger.error(f"Error in search: {traceback.format_exc()}")
            return []

    @staticmethod
    def split_and_clean_text(text: str) -> List[str]:
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
        search_methods: List[SearchMethod],
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
        cache_dir : Optional[str], default None
            Directory for caching search results
        cache_dir_suffix : Optional[str], default None
            Suffix to append to cache directory. Required if cache_dir is specified
        search_methods : List[SearchMethod], default [SearchMethod.TFIDF, SearchMethod.SENTENCE_TRANSFORMER]
            List of search methods to use. These get linked to the corresponding SearchEngine classes.
        init_arguments : Optional[Dict[str, Dict]], default None
            Dictionary containing initialization arguments for each SearchEngine
            Example: {SearchMethod.TFIDF: {"ngram_range": (1, 2)}}
        """
        self.documents = documents
        self.cache_dir = cache_dir
        self.cache_dir_suffix = cache_dir_suffix
        self.search_methods: List[SearchMethod] = search_methods
        self.engines: Dict[SearchMethod, SearchEngine] = {}
        self.engine_init_arguments = init_arguments or {}
        self._initialize_engines()

    def _initialize_engines(self) -> None:
        """Initialize search engines based on specified methods and weights."""
        for method in self.search_methods:
            engine_kwargs = {
                "documents": self.documents,
                "cache_dir": self.cache_dir,
                "cache_dir_suffix": self.cache_dir_suffix or "",
            }

            engine_kwargs.update(self.engine_init_arguments.get(method, {}))

            if method == SearchMethod.TFIDF:
                engine = TFIDFSearchEngine(**engine_kwargs)
            elif method == SearchMethod.SENTENCE_TRANSFORMER:
                engine = SentenceTransformerSearchEngine(**engine_kwargs)
            elif method == SearchMethod.HASHING:
                engine = HashingVectorizerSearchEngine(**engine_kwargs)
            else:
                raise ValueError(f"Unknown search method: {method}")

            self.engines[method] = engine
            engine.create_index()

    def search(
        self,
        query: str,
        top_k: int = 5,
        method_weights: Optional[Dict[SearchMethod, float]] = None,
    ) -> pd.DataFrame:
        """
        Perform ensemble search across all initialized engines and return detailed method scores.

        Parameters:
        -----------
        query : str
            Search query string
        top_k : int, default 5
            Number of top results to return
        method_weights : Optional[Dict[SearchMethod, float]], default None
            Dictionary containing weights for each search method
            Example: {SearchMethod.TFIDF: 0.5, SearchMethod.SENTENCE_TRANSFORMER: 0.5}

        Returns:
        --------
        pd.DataFrame
            DataFrame containing the search results with columns:
            - document info: Information about a given document, like title, name, etc.
            - document: Documentation text
            - method: Search method used
            - score: Raw similarity score
        """
        index_name = "document_name"
        all_results = []
        method_weights = method_weights or {}

        for method, engine in self.engines.items():
            weight = method_weights.get(method, 0)
            # Skip search if weight is 0 or negative
            if weight <= 0:
                continue

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

        # Get top-k unique documents based on aggregated score
        top_documents = (
            df.sort_values("aggregated_score", ascending=False)
            .drop_duplicates(index_name)
            .head(top_k)[index_name]
            .tolist()
        )

        # Filter df to only include top documents
        df_filtered = df[df[index_name].isin(top_documents)]

        # Pivot the scores for each method
        df_methods = df_filtered.pivot(index=index_name, columns="method", values="score").reset_index()

        # Get the aggregated scores for the top documents
        df_agg = df_filtered[["document_name", "document", "aggregated_score"]].drop_duplicates()

        # Merge the method scores with aggregated score
        final_df = df_methods.merge(df_agg, on=index_name).sort_values("aggregated_score", ascending=False)

        # Reorder columns
        method_columns = [
            col for col in final_df.columns if col not in ["document_name", "document", "aggregated_score"]
        ]
        column_order = ["document_name", "document"] + method_columns + ["aggregated_score"]
        final_df = final_df[column_order]

        return final_df

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
        from owlsight.rag.python_lib_search import LibraryInfoExtractor

        context_parts = []

        for _, row in results.iterrows():
            # Get object from full path
            try:
                obj = LibraryInfoExtractor.import_from_string(row["document_name"])
                signature = _get_signature(obj)
            except Exception as e:
                logger.warning(f"Error getting object info: {str(e)}")
                signature = ""

            # Format header with name, signature, and score
            header = f"{row['document_name']}{signature}"

            # Build context entry
            entry = ["=" * 80, header, "-" * 40, "Documentation:", row["document"].strip(), "\n"]

            context_parts.append("\n".join(entry))

        # Combine all entries
        return "\n".join(context_parts)
