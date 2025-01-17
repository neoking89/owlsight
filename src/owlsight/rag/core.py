import re
from typing import Dict, List, Optional, Literal
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from owlsight.rag.custom_classes import CacheMixin, SearchMethod, SearchResult
from owlsight.rag.helper_functions import _get_signature
from owlsight.utils.deep_learning import get_best_device
from owlsight.utils.helper_functions import check_invalid_input_parameters
from owlsight.utils.logger import logger



SENTENCETRANSFORMER_DEFAULT_MODEL = "Alibaba-NLP/gte-base-en-v1.5"


def search_documents(
    query: str,
    documents: Dict[str, str],
    top_k: int = 20,
    tfidf_weight: float = 0.3,
    sentence_transformer_weight: float = 0.7,
    sentence_transformer_model: str = SENTENCETRANSFORMER_DEFAULT_MODEL,
    cache_dir: Optional[str] = None,
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
    cache_dir : Optional[str],
        Directory for caching the embeddings and documentation
    cache_dir_suffix : Optional[str],
        Suffix to add to the cache directory.
        This needs to be specified if cache_dir is provided.
        If specified, the cache directory will be cache_dir/cache_dir_suffix


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


class TFIDFSearchEngine(SearchEngine, CacheMixin):
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


class HashingVectorizerSearchEngine(SearchEngine, CacheMixin):
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


class SentenceTransformerSearchEngine(SearchEngine, CacheMixin):
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
        """
        Initialize the Sentence Transformer search engine.

        Parameters:
        -----------
        documents : Dict[str, str]
            Dictionary containing document names and content
        model_name : str
            Sentence Transformer model name
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
        """
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
            Dictionary containing initialization arguments for each SearchEngine
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

            engine_kwargs = {
                "documents": self.documents,
                "cache_dir": self.cache_dir,
                "cache_dir_suffix": self.cache_dir_suffix or "",
            }

            if method == SearchMethod.TFIDF:
                engine_kwargs.update(self.engine_init_arguments.get(SearchMethod.TFIDF, {}))
                engine = TFIDFSearchEngine(**engine_kwargs)
            elif method == SearchMethod.SENTENCE_TRANSFORMER:
                engine_kwargs.update(self.engine_init_arguments.get(SearchMethod.SENTENCE_TRANSFORMER, {}))
                engine = SentenceTransformerSearchEngine(**engine_kwargs)
            elif method == SearchMethod.HASHING:
                engine_kwargs.update(self.engine_init_arguments.get(SearchMethod.HASHING, {}))
                engine = HashingVectorizerSearchEngine(**engine_kwargs)
            else:
                raise ValueError(f"Unknown search method: {method}")
            
            self.engines[method] = engine
            engine.create_index()

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Perform ensemble search across all initialized engines and return detailed method scores."""
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
        # Get method columns (they'll be between document and aggregated_score)
        method_columns = [
            col for col in final_df.columns if col not in ["document_name", "document", "aggregated_score"]
        ]

        # Create final column order
        column_order = ["document_name", "document"] + method_columns + ["aggregated_score"]

        # Reorder columns
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
                signature = "(Unable to retrieve signature)"

            # Format header with name, signature, and score
            header = f"{row['document_name']}{signature}"
            # score_info = f"(Relevance Score: {row['score']:.3f})"

            # Build context entry
            entry = ["=" * 80, header, "-" * 40, "Documentation:", row["document"].strip(), "\n"]

            context_parts.append("\n".join(entry))

        # Combine all entries
        return "\n".join(context_parts)


# if __name__ == "__main__":
#     import time

#     start = time.time()
#     documents = PythonDocumentationProcessor.get_documents("pandas", cache_dir=".rag_cache")
#     results = search_documents(
#         documents=documents,
#         query="create DataFrame from list",
#         top_k=700,
#         cache_dir=".rag_cache",
#         cache_dir_suffix="pandas",
#     )
#     print(results)
#     print(f"Time taken: {time.time() - start:.2f} seconds")
