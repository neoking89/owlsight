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


class PythonLibSearcher:
    """
    A singleton class for searching Python library documentation with caching capabilities.
    Maintains document and engine caches throughout the owlsight session.
    """

    _instance = None
    _document_cache = {}  # Cache for library documents
    _engine_cache = {}  # Cache for search engines, max size 2

    def _add_to_engine_cache(self, key, value, max_size=5):
        """
        Add a value to the engine cache. Cache size is constrained to a limited number of items.
        """
        if len(self._engine_cache) >= max_size:
            # remove the oldest item
            self._engine_cache.pop(next(iter(self._engine_cache)))
        self._engine_cache[key] = value

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialize only once
        if not hasattr(self, "_initialized"):
            self._initialized = True

    def search(
        self,
        library: str,
        query: str,
        top_k: int = 5,
        cache_dir: Optional[str] = None,
        as_context: bool = True,
        tfidf_weight: float = 1.0,
        sentence_transformer_weight: float = 0.0,
        sentence_transformer_model: str = SENTENCETRANSFORMER_DEFAULT_MODEL,
    ) -> Union[pd.DataFrame, str]:
        """
        Search Python library documentation with caching for documents and search engines.

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
        as_context : bool, default True
            If True, returns formatted context string instead of DataFrame
        tfidf_weight : float, default 1.0
            Weight for the TFIDF search method
        sentence_transformer_weight : float, default 0.0
            Weight for the Sentence Transformer search method
        sentence_transformer_model : str, default "Alibaba-NLP/gte-base-en-v1.5"
            Sentence Transformer model to use

        Returns:
        --------
        Union[pd.DataFrame, str]
            If as_context is True, returns formatted context string
            Otherwise returns DataFrame with search results
        """
        # Get documents from cache or load them
        if library not in self._document_cache:
            self._document_cache[library] = PythonDocumentationProcessor.get_documents(library, cache_dir=cache_dir)
        documents = self._document_cache[library]

        # Configure search methods
        methods_weights = {
            SearchMethod.TFIDF: tfidf_weight,
            SearchMethod.SENTENCE_TRANSFORMER: sentence_transformer_weight,
        }

        # Get or create search engine
        engine_key = self._get_engine_key(library, sentence_transformer_model)
        engine, methods_weights = self._get_or_create_engine(
            engine_key, documents, methods_weights, cache_dir, library, sentence_transformer_model
        )
        results = engine.search(query, top_k=top_k, method_weights=methods_weights)
        results["document_name"] = results["document_name"].apply(lambda x: f"{library}.{x}")

        if as_context:
            return engine.generate_context(results)

        return results

    def clear_cache(self, library: Optional[str] = None):
        """
        Clear the document and engine caches.

        Parameters:
        -----------
        library : Optional[str]
            If provided, only clear caches for the specified library.
            If None, clear all caches.
        """
        if library is None:
            self._document_cache.clear()
            self._engine_cache.clear()
        else:
            if library in self._document_cache:
                del self._document_cache[library]

            # Remove any engine that was created for this library
            keys_to_remove = [key for key in self._engine_cache.keys() if key.startswith(f"{library}_")]
            for key in keys_to_remove:
                del self._engine_cache[key]

    def _get_engine_key(self, library: str, model: str) -> str:
        """Generate a unique cache key for the search engine configuration."""
        return f"{library}_{model}"

    def _create_search_engine(
        self,
        documents: Dict[str, str],
        methods_weights: Dict[SearchMethod, float],
        cache_dir: str,
        library: str,
        sentence_transformer_model: str,
    ) -> Tuple[EnsembleSearchEngine, Dict[SearchMethod, float]]:
        try:
            engine = EnsembleSearchEngine(
                documents=documents,
                search_methods=list(methods_weights.keys()),
                cache_dir=cache_dir,
                cache_dir_suffix=library,
                init_arguments={
                    SearchMethod.SENTENCE_TRANSFORMER: {
                        "pooling_strategy": "mean",
                        "model_name": sentence_transformer_model,
                        "batch_size": 64,
                    }
                },
            )
            return engine, methods_weights
        except Exception as e:
            logger.error(
                f"Failed to load sentence transformer model '{sentence_transformer_model}'. "
                f"Falling back to TF-IDF only."
            )
            logger.error(str(e))
            # Remove sentence transformer from methods and adjust weights to use only TF-IDF
            methods_weights = {SearchMethod.TFIDF: 1.0}
            engine = EnsembleSearchEngine(
                documents=documents,
                search_methods=[SearchMethod.TFIDF],
                cache_dir=cache_dir,
                cache_dir_suffix=library,
            )
            return engine, methods_weights

    def _get_or_create_engine(
        self,
        engine_key: str,
        documents: Dict[str, str],
        methods_weights: Dict[SearchMethod, float],
        cache_dir: str,
        library: str,
        sentence_transformer_model: str,
    ) -> Tuple[EnsembleSearchEngine, Dict[SearchMethod, float]]:
        if engine_key not in self._engine_cache:
            engine, updated_weights = self._create_search_engine(
                documents, methods_weights, cache_dir, library, sentence_transformer_model
            )
            self._add_to_engine_cache(engine_key, engine)
            return engine, updated_weights

        return self._engine_cache[engine_key], methods_weights


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
