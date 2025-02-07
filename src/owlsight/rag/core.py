import re
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Literal, Union
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


import pandas as pd
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from tqdm import tqdm

from owlsight.rag.constants import SENTENCETRANSFORMER_DEFAULT_MODEL
from owlsight.rag.custom_classes import CacheMixin, SearchMethod, SearchResult
from owlsight.rag.helper_functions import _get_signature
from owlsight.utils.deep_learning import get_best_device
from owlsight.utils.helper_functions import check_invalid_input_parameters
from owlsight.utils.logger import logger


class TextSplitter(ABC):
    """Abstract base class for text splitting strategies."""

    @abstractmethod
    def split_documents(self, documents: Dict[str, str], **kwargs) -> Dict[str, str]:
        """Split documents according to the strategy's implementation.

        Parameters
        ----------
        documents : Dict[str, str]
            Dictionary of documents to split, where keys are document names and values are document texts.
        **kwargs
            Additional arguments specific to the splitting strategy

        Returns
        -------
        Dict[str, str]
            Dictionary with new document names as keys and chunks as values.
        """
        pass


class SentenceTextSplitter(TextSplitter):
    """Split text into chunks based on sentences."""

    def __init__(self, n_sentences: int = 3, n_overlap: int = 0):
        """
        Parameters
        ----------
        n_sentences : int, default=3
            Number of sentences per chunk
        n_overlap : int, default=0
            Number of sentences to overlap between chunks
        """
        if n_overlap >= n_sentences:
            raise ValueError("n_overlap must be less than n_sentences")
        self.n_sentences = n_sentences
        self.n_overlap = n_overlap

    @staticmethod
    def split_text_in_sentences(text: str) -> List[str]:
        """Split a longer text into sentences, while keeping account edgecases."""
        text = " " + text.strip() + "  "
        text = text.replace("\n", " ")

        # A small set of "title" abbreviations commonly found
        prefixes = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|St|Mw|Hr)"

        # A few corporate suffixes or honorifics
        suffixes = r"(Inc|Ltd|Jr|Sr|Co)"

        # Some possible expansions: "Mt" for "Mount", "Sen" for "Senator", etc.

        # Common acronym pattern like "U.S.A." or "E.U."
        acronyms = r"([A-Z][.][A-Z][.](?:[A-Z][.])?)"

        # Recognize decimal numbers "5.5" => "5<prd>5"
        digits = r"([0-9])"

        # A small selection of top-level domains (TLDs). Expand as needed:
        # we handle .co.uk, .com, .org, .edu, .io, .net, .gov, .fr, .de, .es, etc.
        websites = r"[.](com|org|net|io|gov|edu|co\.uk|de|fr|es|it|nl|ru|ch|pt|pl|cz|me|be)"

        # Matches multiple periods (could be ellipses, etc.)
        multiple_periods = r"\.{2,}"

        # 1. Protect common abbreviations by rewriting them with <prd>
        text = re.sub(rf"{prefixes}[.]", r"\1<prd>", text)

        # 2. Protect websites / domains: something.com => something<prd>com
        text = re.sub(websites, r"<prd>\1", text)

        # 3. Protect decimals: 3.14 => 3<prd>14
        text = re.sub(rf"{digits}[.]{digits}", r"\1<prd>\2", text)

        # 4. Handle multiple dots: "..." => replace with <prd><prd><prd> + <stop> to mark the end
        #    so we don't inadvertently break them up. e.g. "..." => "<prd><prd><prd><stop>"
        text = re.sub(multiple_periods, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)

        # 5. Special handling for "Ph.D." or similar
        if "Ph.D." in text:
            text = text.replace("Ph.D.", "Ph<prd>D<prd>")

        # 6. Convert single-letter abbreviations: " K. " => " K<prd> "
        #    This ensures we don't treat "K." as end of a sentence if it's an initial
        text = re.sub(rf"\s([A-Za-z])[.]\s", r" \1<prd> ", text)

        # 7. Convert well-known acronyms (U.S.A. => U.S.A<stop>) if followed by capital letter
        #    This helps if the acronym is at end of sentence, though it’s still a simplification
        text = re.sub(rf"{acronyms}\s+([A-Z])", r"\1<stop> \2", text)

        # 8. Convert triple-initial acronyms: "A.B.C." => "A<prd>B<prd>C<prd>"
        text = re.sub(r"([A-Za-z])[.]([A-Za-z])[.]([A-Za-z])[.]", r"\1<prd>\2<prd>\3<prd>", text)
        # 9. Convert double-initial acronyms: "A.B." => "A<prd>B<prd>"
        text = re.sub(r"([A-Za-z])[.]([A-Za-z])[.]", r"\1<prd>\2<prd>", text)

        # 10. Protect suffixes: "Co." => "Co<prd>"
        #     If it's "Company Co. She said", we do "Company Co<prd> She said"
        text = re.sub(rf"\s{suffixes}[.]\s", r" \1<prd> ", text)
        text = re.sub(rf"\s{suffixes}[.]", r" \1<prd>", text)

        # 11. Now, mark the real stops: for . ? !
        #     But watch out for quotes "”" or double quotes right before them
        #     We'll unify them by normalizing some quotes if we want.
        #     This code tries to handle: .", ." , ?" , !", etc.
        #     By reordering them so that the period comes first, or we can do special cases:

        # reorder some ".”" => "”."
        if ".”" in text:
            text = text.replace(".”", "”.")
        if '."' in text:
            text = text.replace('."', '".')
        if '!"' in text:
            text = text.replace('!"', '"!')
        if '?"' in text:
            text = text.replace('?"', '"?')

        # Finally replace the real sentence endings with <stop>
        # So "." => ".<stop>", "?" => "?<stop>", "!" => "!<stop>"
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")

        # Convert all <prd> placeholders back to "."
        text = text.replace("<prd>", ".")

        # 12. Split on <stop>
        sentences = text.split("<stop>")

        # 13. Clean up whitespace, remove empty items
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    @staticmethod
    def split_and_clean_text(text: str) -> List[str]:
        """Split a longer text into sentences and clean them."""
        cleaned_text = text.replace("\n", " ")
        sentences = SentenceTextSplitter.split_text_in_sentences(cleaned_text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def split_documents(self, documents: Dict[str, str], **kwargs) -> Dict[str, str]:
        """Split documents into chunks of n sentences with overlap.

        Parameters
        ----------
        documents : Dict[str, str]
            Dictionary of documents to split, where keys are document names and values are document texts.

        Returns
        -------
        Dict[str, str]
            Dictionary with new document names as keys and sentence chunks as values.
            New document names follow the pattern: [document_name]__split[n]
        """
        split_docs = {}
        for doc_name, doc_text in documents.items():
            # Handle empty documents
            if not doc_text.strip():
                split_docs[f"{doc_name}__split0"] = ""
                continue

            # Split document into sentences
            sentences = self.split_and_clean_text(doc_text)

            # Handle documents with no proper sentences
            if not sentences:
                split_docs[f"{doc_name}__split0"] = doc_text
                continue

            # If document is shorter than chunk size, keep it as one chunk
            if len(sentences) <= self.n_sentences:
                split_docs[f"{doc_name}__split0"] = " ".join(sentences)
                continue

            # Create chunks with overlap
            chunk_idx = 0
            start_idx = 0

            while start_idx + self.n_overlap < len(sentences):
                # Get chunk of n_sentences (or remaining sentences)
                end_idx = min(start_idx + self.n_sentences, len(sentences))
                chunk = sentences[start_idx:end_idx]

                # Create chunk if we have enough sentences for overlap
                chunk_text = " ".join(chunk)
                new_doc_name = f"{doc_name}__split{chunk_idx}"
                split_docs[new_doc_name] = chunk_text
                chunk_idx += 1

                # Move start index forward by (n_sentences - n_overlap)
                start_idx += self.n_sentences - self.n_overlap

        return split_docs


class SemanticTextSplitter(TextSplitter):
    """Split text into chunks based on semantic similarity breakpoints."""

    def __init__(
        self,
        model_name: Optional[Union[str, SentenceTransformer]] = None,
        window_size: int = 1,
        percentile: float = 0.90,
        device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_name : str
            Name of the SentenceTransformer model if no model provided
        window_size : int
            Number of neighbor sentences to include in embedding context
        percentile : float
            Distance percentile threshold for breakpoints (0.0 to 1.0)
        device : Optional[str], default None
            Device to use for Sentence Transformer model
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            SentenceTransformer = None

        if SentenceTransformer is None:
            raise ImportError("Please install `sentence-transformers` to use SemanticTextSplitter.")

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, trust_remote_code=True, device=device) if model_name else None
        self.window_size = window_size
        self.percentile = percentile * 100
        self.device = device

    def split_documents(self, documents: Dict[str, str], show_progress_bar: bool = True, **kwargs) -> Dict[str, str]:
        """Split documents using semantic breakpoint detection."""
        final_results = {}

        for doc_name, doc_text in documents.items():
            doc_text = doc_text.strip()
            if not doc_text:
                final_results[f"{doc_name}__split0"] = ""
                continue

            sentences = SentenceTextSplitter.split_and_clean_text(doc_text)
            if not sentences:
                final_results[f"{doc_name}__split0"] = doc_text
                continue

            if len(sentences) == 1:
                final_results[f"{doc_name}__split0"] = sentences[0]
                continue

            # Create windowed sentences
            windowed_sentences = [
                " ".join(sentences[max(0, i - self.window_size) : i + self.window_size + 1])
                for i in range(len(sentences))
            ]

            # Generate embeddings
            embeddings = self._model.encode(
                windowed_sentences, convert_to_numpy=True, show_progress_bar=show_progress_bar
            )

            # Calculate adjacency distances
            distances = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i + 1].reshape(1, -1))[0][0]
                distances.append(1 - sim)

            # Determine breakpoints
            breakpoints = []
            if distances:
                threshold = np.percentile(distances, self.percentile)
                breakpoints = [i for i, d in enumerate(distances) if d > threshold]

            # Create chunks
            chunks = self._create_chunks(sentences, breakpoints)

            # Store results
            for idx, chunk in enumerate(chunks):
                final_results[f"{doc_name}__split{idx}"] = chunk

        return final_results

    def set_model(self, model: Union[str, SentenceTransformer]):
        """Set the model for the splitter."""
        self._model = (
            model
            if isinstance(model, SentenceTransformer)
            else SentenceTransformer(model, trust_remote_code=True, device=self.device)
        )

    def _create_chunks(self, sentences: List[str], breakpoints: List[int]) -> List[str]:
        """Split sentences into chunks based on breakpoints."""
        chunks = []
        start = 0

        for bp in breakpoints:
            end = bp + 1  # breakpoint index is between sentences
            chunks.append(" ".join(sentences[start:end]))
            start = end

        # Add remaining sentences
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))

        return chunks


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
    text_splitter : Optional[TextSplitter], default=None
        Strategy for splitting documents into chunks. If None, no splitting is done.
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
    - Supports custom text splitting strategies through the TextSplitter interface

    Examples
    --------
    >>> docs = {
    ...     "doc1": "Python is a programming language",
    ...     "doc2": "Machine learning is fascinating"
    ... }
    >>> # Using default sentence-based splitting
    >>> splitter = SentenceTextSplitter(n_sentences=3, n_overlap=1)
    >>> searcher = DocumentSearcher(docs, text_splitter=splitter, cache_dir="document_cache", cache_dir_suffix="programming")
    >>> results = searcher.search("python programming", top_k=3)
    """

    def __init__(
        self,
        documents: Dict[str, str],
        sentence_transformer_model: str = SENTENCETRANSFORMER_DEFAULT_MODEL,
        sentence_transformer_batch_size: int = 64,
        text_splitter: Optional[TextSplitter] = None,
        cache_dir: Optional[str] = None,
        cache_dir_suffix: Optional[str] = None,
    ) -> None:
        self.documents = documents
        self.cache_dir = cache_dir
        self.cache_dir_suffix = cache_dir_suffix
        self.text_splitter = text_splitter

        self._handle_cache_and_documents()

        self.sentence_transformer_model = sentence_transformer_model
        self.sentence_transformer_batch_size = sentence_transformer_batch_size
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
        """
        Load a DocumentSearcher instance from earlier cached documents and embeddings.

        Parameters
        ----------
        cache_dir : str
            Directory containing .pkl files
        cache_dir_suffix : str
            Suffix for cache directory name. This is used to identify the correct cache files.
        **init_kwargs
            Additional arguments to pass to the DocumentSearcher constructor
        """
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

        pkl_files = [f[:-4] for f in os.listdir(cache_dir) if f.endswith(".pkl")]
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in cache directory: {cache_dir}")

        if not any(map(lambda file: cache_dir_suffix == file.split("__")[1], pkl_files)):
            raise FileNotFoundError(
                f"No cached files found in dir {cache_dir} with cache_dir_suffix: {cache_dir_suffix}"
            )

        init_cls = cls(
            documents={},  # documents get loaded from cache, so we pass an empty dict
            cache_dir=cache_dir,
            cache_dir_suffix=cache_dir_suffix,
            **init_kwargs,
        )

        return init_cls

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

    def _create_unique_cache_dir_suffix(self) -> str:
        """Create unique cache names based on instance configuration"""
        if self.text_splitter:
            # Get the class name and parameters for the text splitter
            splitter_config = f"{self.text_splitter.__class__.__name__}"
            for key, value in self.text_splitter.__dict__.items():
                if key.startswith("_"):
                    continue
                safe_value = self._clean_path_for_cache(value)
                splitter_config += f"__{key}={safe_value}"
            self.cache_dir_suffix = f"{self.cache_dir_suffix}__{splitter_config}"
        return self.cache_dir_suffix

    def _clean_path_for_cache(self, value: Any) -> str:
        """Clean a value for use in cache path by replacing problematic characters."""
        return str(value).replace('\\', '-').replace('/', '-')

    def _handle_cache_and_documents(self):
        if not self.documents and not self.cache_dir:
            raise ValueError("documents must not be empty if cache_dir is not provided")

        if self.cache_dir and self.cache_dir_suffix:
            self.cache_dir_suffix = self._create_unique_cache_dir_suffix()
            cache_mixin = CacheMixin(
                cache_dir=self.cache_dir,
                cache_dir_suffix=f"cache_dir_suffix={self.cache_dir_suffix}",
            )
            if cache_mixin.get_full_cache_path().exists() and not self.documents:
                self.documents = cache_mixin.load_data()
            elif self.text_splitter:
                self.documents = self.text_splitter.split_documents(self.documents)
                cache_mixin.save_data(self.documents)
        elif self.text_splitter:
            self.documents = self.text_splitter.split_documents(self.documents)


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
                    sentences = SentenceTextSplitter.split_and_clean_text(text)
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

        except Exception:
            logger.error(f"Error in search: {traceback.format_exc()}")
            return []

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
