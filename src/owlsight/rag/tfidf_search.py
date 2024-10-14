import importlib
import inspect
import pkgutil
from typing import List, Dict, Generator, Any, Union, Literal
import time
import re

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys

sys.path.append("src")
from owlsight.utils.deep_learning import get_best_device
from owlsight.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def get_context_for_library(
    library_name: str,
    query: str,
    top_k: int = 3,
    method: Literal["cosine", "faiss", "sentence-transformer"] = "cosine",
    get_results_only: bool = False,
) -> Union[str, List[Dict]]:
    """
    Searches for the top-k most relevant functions/classes in library documentation based on a query.

    Parameters:
        library_name (str): The name of the library to search in.
        query (str): The search query.
        top_k (int): The number of top results to return.
        method (str): The search method to use ("cosine""faiss").
        get_results_only (bool): If True, only the searchresults (dict) will be returned instead of the full context.

    Returns:
        str: The context (documentation) of the top-k search results for the given library and query.
    """
    search_engine: BaseLibrarySearch = _get_search_engine(method, library_name)
    search_engine.create_index()
    results = search_engine.search(query, top_k)
    if get_results_only:
        return pd.DataFrame.from_dict(results)

    context = search_engine.generate_context(results)
    return context


class BaseLibrarySearch:
    def __init__(self, library_name: str, create_cache: bool = False):
        """
        Base class for searching in documentation of a Python library.

        Parameters:
            library_name (str): The name of the library to search in.
            create_cache (bool): Whether to create a cache for storing the index.
            As default, has name
        """
        self.target_library_name = library_name
        self.target_library = importlib.import_module(library_name)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")

        if create_cache:
            cache_name = f"{library_name}__{self.__class__.__name__}.pkl"

        self.tfidf_matrix = None
        self.target_library_info = {}
        self.corpus = []

    def extract_library_info(self) -> Generator[tuple, None, None]:
        def explore_module(module, prefix="") -> Generator[tuple, None, None]:
            if not hasattr(module, "__path__"):
                return

            for _, name, is_pkg in pkgutil.iter_modules(module.__path__):
                full_name = f"{prefix}.{name}" if prefix else name

                # Skip test modules
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

    def create_index(self):
        logger.info(
            f"Extracting library information from {self.target_library_name}..."
        )
        for name, info in self.extract_library_info():
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

    def _extract_info_from_module(
        self, module, prefix=""
    ) -> Generator[tuple, None, None]:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
                doc = inspect.getdoc(obj)
                if doc:
                    full_name = f"{prefix}.{name}" if prefix else name
                    yield full_name, {"doc": doc, "obj": obj}

    def _create_tfidf_index(self):
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        except Exception as e:
            logger.error(f"Error creating TF-IDF index: {str(e)}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        raise NotImplementedError("Search method not implemented.")

    def __str__(self):
        return f"{self.__class__.__name__}(library={self.target_library_name})"


class CosineSimilaritySearch(BaseLibrarySearch):
    """
    Search engine using cosine similarity for searching in a Python library.
    """

    def create_index(self):
        super().create_index()
        self._create_tfidf_index()

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


class FaissSearch(BaseLibrarySearch):
    def __init__(self, library_name: str):
        """
        Search engine using FAISS for searching in a Python library.
        """
        from faiss import IndexFlatL2

        self.index_func = IndexFlatL2
        super().__init__(library_name)
        self.index = None  # FAISS index

    def create_index(self):
        super().create_index()
        self._create_tfidf_index()

        if self.tfidf_matrix is not None:
            self.index = self.index_func(
                self.tfidf_matrix.shape[1]
            )  # L2 distance (Euclidean)
            self.index.add(self.tfidf_matrix.toarray().astype(np.float32))

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None:
            return []

        query_vec = (
            self.tfidf_vectorizer.transform([query]).toarray().astype(np.float32)
        )
        _, top_indices = self.index.search(query_vec, top_k)

        results = []
        for idx in top_indices[0]:
            name = list(self.target_library_info.keys())[idx]
            info = self.target_library_info[name]
            results.append(
                {
                    "name": name,
                    "score": float(
                        np.dot(query_vec, self.tfidf_matrix[idx].toarray().T)[0][0]
                    ),  # Dot product score
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
            model_name (str): The name of the Sentence Transformer model to use.
            device (str): The device to run the model on (e.g. "cuda" or "cpu").
        """

        from sentence_transformers import SentenceTransformer, util

        self.SentenceTransformer = SentenceTransformer
        self.util = util
        super().__init__(library_name)
        self.model_name = model_name
        self.device = get_best_device() if device is None else device
        self.model = None
        self.embeddings = None

    def create_index(self):
        super().create_index()

        # Initialize the model
        self.model = self.SentenceTransformer(self.model_name, device=self.device)

        # Preprocess the corpus: split into sentences
        self.corpus = [split_and_clean_text(text) for text in self.corpus]

        # Create embeddings for the library documentation
        self.embeddings = []

        # Add tqdm to show the progress bar
        for text in tqdm(
            self.corpus,
            desc="Generating embeddings",
            unit="document ",
            total=len(self.corpus),
        ):
            # Encode the sentences in the document and append to the embeddings list
            sentence_embeddings = self.model.encode(text, convert_to_tensor=True)

            # Average the sentence embeddings to get a document embedding
            document_embedding = torch.mean(sentence_embeddings, dim=0)
            self.embeddings.append(document_embedding)

        # Convert list to tensor
        self.embeddings = torch.stack(self.embeddings)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            return []

        # Generate embedding for the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarities
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


def _get_search_engine(method: str, library_name: str) -> BaseLibrarySearch:
    """
    Returns the search engine based on the method.

    Available methods:
        - "cosine": Cosine similarity search
        - "faiss": FAISS search
        - "sentence-transformer": Sentence Transformer search
    """
    search_methods = {
        "cosine": CosineSimilaritySearch,
        "faiss": FaissSearch,
        "sentence-transformer": SentenceTransformerSearch,
    }

    try:
        search_engine_class = search_methods[method]
        return search_engine_class(library_name)
    except KeyError:
        raise ValueError(
            f"Unknown search method: {method}. Available methods: {', '.join(search_methods.keys())}"
        )
    except ImportError as e:
        logger.warning(f"{e.name} not available, falling back to cosine similarity.")
        return CosineSimilaritySearch(library_name)


def split_and_clean_text(text: str) -> list:
    # Step 1: Remove newlines
    cleaned_text = text.replace("\n", " ")

    # Step 2: Split into sentences based on common sentence end punctuation
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", cleaned_text)

    # Step 3: Strip leading and trailing spaces from sentences
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences


if __name__ == "__main__":
    for method in ["faiss", "cosine", "sentence-transformer"]:
        print(f"Using search method: {method}")
        start = time.time()
        results = get_context_for_library(
            "pandas",
            "How to merge 2 dataframes?",
            method=method,
            get_results_only=True,
            top_k=10,
        )
        print(results)
        end = time.time()
        print(f"Time taken: {end - start:.2f} seconds\n")
