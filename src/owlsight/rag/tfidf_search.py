from typing import List, Dict, Generator, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import importlib
import inspect
import pkgutil

from owlsight.utils.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)

def get_context_for_library(library_name: str, query: str, top_k: int = 3) -> str:
    """
    Searches for the top-k most relevant functions/classes in a library based on a query.

    Parameters:
        library_name (str): The name of the library to search in.
        query (str): The search query.
        top_k (int): The number of top results to return.

    Returns:
        str: The context (documentation) of the top-k search results for given library and query.
    """
    tfidf_search = TfidfLibrarySearch(library_name)
    tfidf_search.create_index()
    results = tfidf_search.search(query, top_k)
    context = tfidf_search.generate_context(results)
    return context


class TfidfLibrarySearch:
    def __init__(self, library_name: str):
        self.library_name = library_name
        self.library = importlib.import_module(library_name)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.library_info = {}
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
            yield from explore_module(self.library)
        except Exception as e:
            logger.error(f"Error exploring {self.library_name}: {str(e)}")

    def _extract_info_from_module(
        self, module, prefix=""
    ) -> Generator[tuple, None, None]:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
                doc = inspect.getdoc(obj)
                if doc:
                    full_name = f"{prefix}.{name}" if prefix else name
                    yield full_name, {"doc": doc, "obj": obj}

    def create_index(self):
        logger.info(f"Extracting library information from {self.library_name}...")
        for name, info in self.extract_library_info():
            try:
                self.library_info[f"{self.library_name}.{name}"] = info
                self.corpus.append(info["doc"])
            except Exception as e:
                logger.error(f"Error extracting info from {name}: {str(e)}")

        if not self.corpus:
            logger.warning(f"No documentation found for {self.library_name}")
            return

        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus)
        except Exception as e:
            logger.error(f"Error creating TF-IDF index: {str(e)}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.tfidf_matrix is None:
            return []

        query_vec = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            name = list(self.library_info.keys())[idx]
            info = self.library_info[name]
            results.append(
                {
                    "name": name,
                    "score": float(similarities[idx]),
                    "doc": info["doc"],
                    "obj": info["obj"],
                }
            )

        return results

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


# Example usage:
if __name__ == "__main__":
    context = get_context_for_library("pandas", "How to read a CSV file?")
    print(context)
