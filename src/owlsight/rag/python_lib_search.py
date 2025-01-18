from typing import Any, Dict, Generator, Tuple, Optional, Union
import importlib
import inspect
import pkgutil

import pandas as pd

from owlsight.rag.core import EnsembleSearchEngine
from owlsight.rag.custom_classes import CacheMixin, SearchMethod
from owlsight.rag.constants import SENTENCETRANSFORMER_DEFAULT_MODEL
from owlsight.utils.logger import logger


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


def search_python_libs(
    library: str,
    query: str,
    top_k: int = 5,
    cache_dir: Optional[str] = None,
    return_context: bool = True,
    tfidf_weight: float = 1.0,
    sentence_transformer_weight: float = 0.0,
    sentence_transformer_model: str = SENTENCETRANSFORMER_DEFAULT_MODEL,
    sentence_transformer_batch_size: int = 64,
) -> Union[pd.DataFrame, str]:
    """
    Get search results for Python library documentation with optional formatted context.
    This context can be added to the output of a chatbot.
    Or it can be used as search engine within python libraries.

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
    tfidf_weight : float, default 1.0
        Weight for the TFIDF search method
    sentence_transformer_weight : float, default 0.0
        Weight for the Sentence Transformer search method.
        If more than 0, next to TFIDF, the sentence transformer method will be used.
        NOTE: Using this can lead to more accurate search results, but also much slower processing since the embeddings need to be computed first.
    sentence_transformer_model : str, default "Alibaba-NLP/gte-base-en-v1.5"
        Sentence Transformer model to use
    sentence_transformer_batch_size : int, default 64
        Batch size to use for Sentence Transformer embeddings

    Returns:
    --------
    Union[pd.DataFrame, str]
        If return_context is True, returns formatted context string
        Otherwise returns DataFrame with search results
    """
    documents = PythonDocumentationProcessor.get_documents(library, cache_dir=cache_dir)
    methods_weights = {
        SearchMethod.TFIDF: tfidf_weight,
        SearchMethod.SENTENCE_TRANSFORMER: sentence_transformer_weight,
    }

    engine = EnsembleSearchEngine(
        documents=documents,
        methods_weights=methods_weights,
        cache_dir=cache_dir,
        cache_dir_suffix=library,
        init_arguments={
            SearchMethod.SENTENCE_TRANSFORMER: {
                "pooling_strategy": "mean",
                "model_name": sentence_transformer_model,
                "batch_size": sentence_transformer_batch_size,
            }
        },
    )

    results = engine.search(query, top_k=top_k)
    results["document_name"] = results["document_name"].apply(lambda x: f"{library}.{x}")

    if return_context:
        return engine.generate_context(results)

    return results


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

    def extract_library_info(self) -> Dict[str, str]:
        """Extract documentation from the library."""
        if self.cache_dir:
            cached_data = self.load_data()
            if cached_data is not None:
                return cached_data

        unique_docs = {}
        # add documentation as key to keep it unique
        for full_name, doc_info in self._extract_library_info_as_generator():
            if doc_info and "doc" in doc_info:
                unique_docs[doc_info["doc"]] = full_name

        # afterwards, reverse the key-value pairs to have the object name as key
        unique_docs = {name: doc for doc, name in unique_docs.items() if doc}

        if self.cache_dir:
            self.save_data(unique_docs)

        return unique_docs

    def _extract_library_info_as_generator(self) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Extract documentation from the library."""

        def explore_module(module, prefix="", visited=None) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
            if visited is None:
                visited = set()

            # Skip if module has no __path__ or has been visited
            if not hasattr(module, "__path__") or module.__name__ in visited:
                return

            visited.add(module.__name__)

            try:
                module_iter = pkgutil.iter_modules(module.__path__)
            except Exception:
                return

            for _, name, is_pkg in module_iter:
                # Skip test modules and private modules
                if name.startswith("_") or "test" in name.lower():
                    continue

                full_name = f"{prefix}.{name}" if prefix else name

                try:
                    # Try to import the module
                    sub_module = importlib.import_module(f"{module.__name__}.{name}")

                    # Extract info from current module
                    for item in self._extract_info_from_module(sub_module, full_name):
                        yield item

                    # If it's a package, explore it recursively
                    if is_pkg:
                        yield from explore_module(sub_module, full_name, visited)

                except (ImportError, AttributeError, ModuleNotFoundError):
                    # Silently skip problematic imports
                    continue
                except Exception as e:
                    # Log other unexpected errors but continue processing
                    logger.error(f"Unexpected error exploring {full_name}: {str(e)}")
                    continue

        try:
            yield from explore_module(self.library)
        except Exception as e:
            logger.error(f"Error exploring {self.library_name}: {str(e)}")

    def _extract_info_from_module(
        self, module: Any, prefix: str = ""
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """Extract documentation from a specific module."""
        try:
            for name, obj in inspect.getmembers(module):
                try:
                    # Skip private members
                    if name.startswith("_"):
                        continue

                    # Check if the object is part of the library
                    if getattr(obj, "__module__", "").startswith(self.library_name):
                        if inspect.isclass(obj):
                            # If it's a class, include its methods
                            for class_name, class_obj in inspect.getmembers(obj):
                                if inspect.ismethod(class_obj) or inspect.isfunction(class_obj):
                                    doc = inspect.getdoc(class_obj)
                                    if doc:
                                        full_name = (
                                            f"{prefix}.{name}.{class_name}" if prefix else f"{name}.{class_name}"
                                        )
                                        yield full_name, {"doc": doc, "obj": class_obj}
                        elif inspect.isfunction(obj) or inspect.ismethod(obj):
                            doc = inspect.getdoc(obj)
                            if doc:
                                full_name = f"{prefix}.{name}" if prefix else name
                                yield full_name, {"doc": doc, "obj": obj}
                except Exception:
                    continue
        except Exception:
            return
