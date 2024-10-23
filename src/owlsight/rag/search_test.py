# TODO: tokenize the documents to List of List of words before passing to the search engine (?)

import sys

sys.path.append("src")

from typing import List, Dict, Any, Union, Literal
from tqdm import tqdm
import pickle
from pathlib import Path
import importlib
import inspect
import pkgutil
from typing import Generator

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from owlsight.utils.deep_learning import get_best_device
from owlsight.utils.helper_functions import check_invalid_input_parameters
from owlsight.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class BaseSearchEngine:
    def __init__(self, documents: List[str], cache_dir: str = None):
        self.documents = documents
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        raise NotImplementedError("Search method not implemented.")

    def save_data(self, data: Any):
        if self.cache_dir:
            with open(self.cache_filename, "wb") as f:
                pickle.dump(data, f)

    def load_data(self) -> Any:
        if self.cache_dir and self.cache_filename.exists():
            with open(self.cache_filename, "rb") as f:
                return pickle.load(f)
        return None

    @property
    def cache_filename(self) -> Path:
        return Path(self.cache_dir) / f"{self.__class__.__name__}.pkl"


class TfidfVectorizerSearch(BaseSearchEngine):
    def __init__(self, documents: List[str], cache_dir: str = None, **tfidf_kwargs):
        super().__init__(documents, cache_dir)
        check_invalid_input_parameters(TfidfVectorizer.__init__, tfidf_kwargs)
        self.tfidf_vectorizer = TfidfVectorizer(**tfidf_kwargs)
        self.tfidf_matrix = None

    def create_index(self):
        cached_data = self.load_data()
        if cached_data is not None:
            self.tfidf_matrix, self.tfidf_vectorizer = cached_data
        else:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
            self.save_data((self.tfidf_matrix, self.tfidf_vectorizer))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.tfidf_matrix is None:
            return []

        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = [{"document": self.documents[idx], "score": float(similarities[idx])} for idx in top_indices]
        return results


class SentenceTransformerSearch(BaseSearchEngine):
    def __init__(
        self,
        documents: List[str],
        model_name: str = "paraphrase-MiniLM-L6-v2",
        device: str = None,
        cache_dir: str = None,
    ):
        from sentence_transformers import SentenceTransformer, util

        super().__init__(documents, cache_dir)
        self.SentenceTransformer = SentenceTransformer
        self.util = util
        self.model_name = model_name
        self.device = get_best_device() if device is None else device
        self.model = None
        self.embeddings = None

    def create_index(self):
        self.model = self.SentenceTransformer(self.model_name, device=self.device)
        self.embeddings = self.load_data()
        if self.embeddings is None:
            self._create_embeddings()
            self.save_data(self.embeddings)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            return []

        query_embedding = self.model.encode(query, convert_to_tensor=True).unsqueeze(0)
        similarities = self.util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_indices = similarities.topk(k=top_k).indices

        results = [{"document": self.documents[idx], "score": float(similarities[idx])} for idx in top_indices]
        return results

    def _create_embeddings(self):
        self.embeddings = [
            self.model.encode(text, convert_to_tensor=True)
            for text in tqdm(self.documents, desc="Generating embeddings")
        ]
        self.embeddings = torch.stack(self.embeddings, dim=0)


class HashingVectorizerSearch(BaseSearchEngine):
    def __init__(self, documents: List[str], cache_dir: str = None, **hashing_kwargs):
        super().__init__(documents, cache_dir)
        check_invalid_input_parameters(HashingVectorizer.__init__, hashing_kwargs)
        self.hash_vectorizer = HashingVectorizer(**hashing_kwargs)
        self.hash_matrix = None

    def create_index(self):
        cached_data = self.load_data()
        if cached_data is not None:
            self.hash_matrix, self.hash_vectorizer = cached_data
        else:
            self.hash_matrix = self.hash_vectorizer.transform(self.documents)
            self.save_data((self.hash_matrix, self.hash_vectorizer))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.hash_matrix is None:
            return []

        query_vec = self.hash_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.hash_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = [{"document": self.documents[idx], "score": float(similarities[idx])} for idx in top_indices]
        return results


class LibraryInfoExtractor:
    def __init__(self, library_name: str):
        self.library_name = library_name
        self.library = importlib.import_module(library_name)

    def extract_library_info(self) -> Generator[tuple, None, None]:
        def explore_module(module, prefix="") -> Generator[tuple, None, None]:
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
                    logger.error(f"Skipping {full_name}: {str(e)}")

        try:
            yield from explore_module(self.library)
        except Exception as e:
            logger.error(f"Error exploring {self.library_name}: {str(e)}")

    def _extract_info_from_module(self, module, prefix="") -> Generator[tuple, None, None]:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
                doc = inspect.getdoc(obj)
                if doc:
                    full_name = f"{prefix}.{name}" if prefix else name
                    yield full_name, {"doc": doc, "obj": obj}


def get_search_engine(method: str, documents: List[str], **kwargs) -> BaseSearchEngine:
    search_methods = {
        "tfidf": TfidfVectorizerSearch,
        "sentence-transformer": SentenceTransformerSearch,
        "hashing": HashingVectorizerSearch,
    }

    try:
        search_engine_class = search_methods[method]
        return search_engine_class(documents, **kwargs)
    except KeyError:
        raise ValueError(f"Unknown search method: {method}. Available methods: {', '.join(search_methods.keys())}")


def ensemble_search(
    query: str,
    documents: List[str],
    top_k: int = 5,
    methods_with_weights: Dict[str, float] = None,
    cache_dir: str = None,
) -> pd.DataFrame:
    results_combined = []

    if methods_with_weights is None:
        raise ValueError("You must provide a dictionary with method names and their corresponding weights.")

    if not all(isinstance(document, str) for document in documents):
        renewed_documents = []
        for doc in documents:
            if not isinstance(doc, str):
                if hasattr(doc, "__doc__"):
                    doc = doc.__doc__
                    renewed_documents.append(doc)
                else:
                    continue
            else:
                renewed_documents.append(doc)

        documents = renewed_documents

    # only leave documents that are strings
    documents = [doc for doc in documents if isinstance(doc, str)]

    assert all(isinstance(document, str) for document in documents), "All documents must be strings."

    # raise TypeError("All documents must be strings.")

    for method, weight in methods_with_weights.items():
        if weight is None or weight <= 0 or weight > 1:
            continue
        search_engine = get_search_engine(method, documents, cache_dir=cache_dir)
        search_engine.create_index()
        results = search_engine.search(query, top_k=top_k)
        df_results = pd.DataFrame(results)
        df_results["method"] = method
        df_results["weighted_score"] = df_results["score"] * weight
        results_combined.append(df_results)

    all_results = pd.concat(results_combined)
    all_results["aggregated_score"] = all_results.groupby("document")["weighted_score"].transform("sum")

    return (
        all_results.sort_values(by="aggregated_score", ascending=False).drop_duplicates(subset=["document"]).head(top_k)
    )


if __name__ == "__main__":
    # Extracting pandas documentation
    library_name = "pandas"
    extractor = LibraryInfoExtractor(library_name)
    documents = [doc for _, doc_info in extractor.extract_library_info() for doc in doc_info.values()]

    # Query
    search_query = "How do I merge 2 dataframes?"

    # Running ensemble search with equal weights for all methods
    methods_with_weights = {
        "tfidf": 1.0,
        "sentence-transformer": 0.0,
        "hashing": 1.0,
    }

    results = ensemble_search(
        query=search_query,
        documents=documents,
        top_k=5,
        methods_with_weights=methods_with_weights,
        cache_dir="cache_dir",
    )

    print(results)
