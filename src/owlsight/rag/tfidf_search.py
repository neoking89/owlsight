import importlib
import inspect
import pkgutil
from typing import List, Dict, Generator, Any, Union, Literal
import re
import pickle
from pathlib import Path

import torch
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# import sys
# sys.path.append("src")
from owlsight.utils.deep_learning import get_best_device
from owlsight.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def get_context_for_library(
    library_name: str,
    query: str,
    top_k: int = 3,
    method: Literal["cosine", "sentence-transformer"] = "cosine",
    get_results_only: bool = False,
    cache_dir: str = None,
) -> Union[str, List[Dict]]:
    """
    Searches for the top-k most relevant functions/classes in library documentation based on a query.

    Parameters:
        library_name (str): The name of the library to search in.
        query (str): The search query.
        top_k (int): The number of top results to return.
        method (str): The search method to use ("cosine", "sentence-transformer").
        get_results_only (bool): If True, only the searchresults (dict) will be returned instead of the full context.
        cache_dir (str): The directory to cache the search data. Useful if generating embeddings (Sentence Transformer) takes long.

    Returns:
        str: The context (documentation) of the top-k search results for the given library and query.
    """
    search_engine: BaseLibrarySearch = get_search_engine(
        method, library_name, cache_dir=cache_dir
    )
    search_engine.create_index()
    results = search_engine.search(query, top_k)
    if get_results_only:
        return pd.DataFrame.from_dict(results)

    context = search_engine.generate_context(results)
    return context


class LibraryInfoExtractor:
    """
    Extracts information from a Python library.
    """

    def __init__(self, library_name: str):
        self.target_library_name = library_name
        self.target_library = importlib.import_module(library_name)

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
            yield from explore_module(self.target_library)
        except Exception as e:
            logger.error(f"Error exploring {self.target_library_name}: {str(e)}")

    def _extract_info_from_module(
        self, module, prefix=""
    ) -> Generator[tuple, None, None]:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
                doc = inspect.getdoc(obj)
                if doc:
                    full_name = f"{prefix}.{name}" if prefix else name
                    yield full_name, {"doc": doc, "obj": obj}


class BaseLibrarySearch:
    """
    Base class for searching in a Python library.
    """

    def __init__(self, library_name: str, cache_dir: str = None):
        self.target_library_name = library_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.target_library_info = {}
        self.corpus = []
        self.extractor = LibraryInfoExtractor(library_name)

    def create_index(self):
        logger.info(
            f"Extracting library information from {self.target_library_name}..."
        )
        for name, info in self.extractor.extract_library_info():
            try:
                self.target_library_info[f"{self.target_library_name}.{name}"] = info
                self.corpus.append(info["doc"])
            except Exception as e:
                logger.error(f"Error extracting info from {name}: {str(e)}")

        if not self.corpus:
            logger.warning(f"No documentation found for {self.target_library_name}")
            return

    def generate_context(self, search_results: List[Dict[str, Any]]) -> str:
        context = ""
        for result in search_results:
            name = result["name"]
            obj = result["obj"]
            doc = result["doc"]

            try:
                signature = str(inspect.signature(obj)) if callable(obj) else ""
            except ValueError:
                signature = "(Unable to retrieve signature)"

            context += f"{name}{signature}\n"
            context += f"Documentation:\n{doc}\n\n"

        return context

    @property
    def cache_filename(self) -> Path:
        if hasattr(self, "model_name"):
            return (
                Path(self.cache_dir)
                / f"{self.target_library_name}__{self.__class__.__name__}__{self.model_name}.pkl"
            )
        return (
            Path(self.cache_dir)
            / f"{self.target_library_name}__{self.__class__.__name__}.pkl"
        )

    def save_data(self, data):
        if self.cache_dir:
            with open(self.cache_filename, "wb") as f:
                pickle.dump(data, f)

    def load_data(self):
        if self.cache_dir and self.cache_filename.exists():
            with open(self.cache_filename, "rb") as f:
                return pickle.load(f)
        return None

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        raise NotImplementedError("Search method not implemented.")


class CosineSimilaritySearch(BaseLibrarySearch):
    def __init__(self, library_name: str, cache_dir: str = None):
        """
        Search engine using cosine similarity for searching in a Python library.

        Parameters:
        ----------
            library_name (str): The name of the library to search in.
            cache_dir (str): The directory to cache the search data.
        """
        super().__init__(library_name, cache_dir)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None

    def create_index(self):
        super().create_index()

        cached_data = self.load_data()
        if cached_data is not None:
            self.tfidf_matrix, self.tfidf_vectorizer = cached_data
        else:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
            self.save_data((self.tfidf_matrix, self.tfidf_vectorizer))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.tfidf_matrix is None:
            return []

        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            name = list(self.target_library_info.keys())[idx]
            info = self.target_library_info[name]
            results.append(
                {
                    "name": name,
                    "score": float(similarities[idx]),
                    "doc": info["doc"],
                    "obj": info["obj"],
                }
            )

        return results


class SentenceTransformerSearch(BaseLibrarySearch):
    def __init__(
        self,
        library_name: str,
        model_name="paraphrase-MiniLM-L6-v2",
        device: str = None,
    ):
        """
        Search engine using Sentence Transformer for searching in a Python library.

        Parameters:
        ----------
            library_name (str): The name of the library to search in.
            model_name (str): The name of the Sentence Transformer model to use.
            device (str): The device to run the model on (e.g. "cuda" or "cpu").
        """

        from sentence_transformers import SentenceTransformer, util

        self.SentenceTransformer = SentenceTransformer
        self.util = util
        self.model_name = model_name
        self.device = get_best_device() if device is None else device
        self.model = None
        self.embeddings = None

    def create_index(self):
        super().create_index()
        self.model = self.SentenceTransformer(self.model_name, device=self.device)

        self.embeddings = self.load_data()
        if self.embeddings is None:
            self.corpus = [split_and_clean_text(text) for text in self.corpus]
            self._create_embeddings()
            self.save_data(self.embeddings)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            return []

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        similarities = self.util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
        top_indices = similarities.topk(k=top_k).indices

        results = []
        for idx in top_indices:
            name = list(self.target_library_info.keys())[idx]
            info = self.target_library_info[name]
            results.append(
                {
                    "name": name,
                    "score": float(similarities[idx]),
                    "doc": info["doc"],
                    "obj": info["obj"],
                }
            )

        return results

    def _create_embeddings(self):
        self.embeddings = []
        for text in tqdm(
            self.corpus, desc="Generating embeddings", total=len(self.corpus)
        ):
            sentence_embeddings = self.model.encode(text, convert_to_tensor=True)
            document_embedding = torch.mean(sentence_embeddings, dim=0)
            self.embeddings.append(document_embedding)
        self.embeddings = torch.stack(self.embeddings)


def split_and_clean_text(text: str) -> list:
    cleaned_text = text.replace("\n", " ")
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", cleaned_text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def get_search_engine(method: str, library_name: str, **kwargs) -> BaseLibrarySearch:
    search_methods = {
        "cosine": CosineSimilaritySearch,
        "sentence-transformer": SentenceTransformerSearch,
    }

    try:
        search_engine_class = search_methods[method]
        return search_engine_class(library_name, **kwargs)
    except KeyError:
        raise ValueError(
            f"Unknown search method: {method}. Available methods: {', '.join(search_methods.keys())}"
        )
    except ImportError as e:
        logger.warning(f"{e.name} not available, falling back to cosine similarity.")
        return CosineSimilaritySearch(library_name, **kwargs)


# if __name__ == "__main__":
#     for method in ["cosine", "sentence-transformer"]:
#         print(f"Using search method: {method}")
#         start = time.time()
#         results = get_context_for_library(
#             "shiny",
#             "How to create a basic dashboard?",
#             method=method,
#             get_results_only=True,
#             top_k=10,
#             cache_dir="rag_cache",
#         )
#         print(results)
#         end = time.time()
#         print(f"Time taken: {end - start:.2f} seconds\n")
