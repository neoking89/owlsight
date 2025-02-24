from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional, TypeVar, Union, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from owlsight.rag.constants import SENTENCETRANSFORMER_DEFAULT_MODEL


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
        text = re.sub(r"\s([A-Za-z])[.]\s", r" \1<prd> ", text)

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


SENCENCETRANSFORMER_TYPE = TypeVar("SentenceTransformer")


class SemanticTextSplitter(TextSplitter):
    """Split text into chunks based on semantic similarity breakpoints."""

    def __init__(
        self,
        model_name: str = SENTENCETRANSFORMER_DEFAULT_MODEL,
        window_size: int = 0,
        percentile: float = 0.90,
        device: Optional[str] = None,
        target_chunk_length: Optional[int] = None,
        sentence_transformer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Splits text documents into semantically coherent chunks using SentenceTransformer embeddings.

        This splitter works by first splitting a document into sentences, then creating a "windowed"
        version of these sentences to incorporate contextual information. It generates embeddings for
        each windowed sentence and computes the cosine distance between adjacent embeddings. A semantic
        breakpoint is determined when the distance exceeds a threshold, which is set as a specific
        percentile of all computed distances. The document is then split at these breakpoints so that
        each chunk represents a segment with relatively homogeneous semantic content.

        Parameters
        ----------
        model_name : str
            The name or identifier of the SentenceTransformer model used for embedding.
            This can be a local path or a model identifier from the Hugging Face Hub.
        window_size : int, default 0
            The number of sentences to consider in each window when calculating similarity scores.
            A larger window size considers more context but may result in larger chunks.
            Ideally, this is kept to 0 when using sentence_transformers to embed every sentence.
        percentile : float, default 0.90
            The percentile threshold for similarity scores when deciding chunk boundaries.
            Higher values create more chunks, lower values create fewer but larger chunks.
        device : Optional[str]
            The device (e.g., 'cpu' or 'cuda') on which to run the SentenceTransformer model.
            If None, automatically selects the best available device.
        target_chunk_length : Optional[int]
            If set, aims to maintain this as the mean chunk length in characters.
            This is a soft target that helps balance chunk sizes while preserving semantic relevance.
        sentence_transformer_kwargs : Optional[Dict[str, Any]], default None
            Additional keyword arguments to pass to the SentenceTransformer constructor

        Attributes
        ----------
        All above parameters are passed to the `SentenceTransformer` constructor.
        _model : Optional[SentenceTransformer]
            An instance of the SentenceTransformer model loaded using the specified model_name.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            SentenceTransformer = None

        if SentenceTransformer is None:
            raise ImportError("Please install `sentence-transformers` to use SemanticTextSplitter.")

        self.model_name = model_name
        # Initialize model with additional kwargs if provided
        model_kwargs = {"trust_remote_code": True, "device": device}
        if sentence_transformer_kwargs:
            model_kwargs.update(sentence_transformer_kwargs)
        self._model = SentenceTransformer(model_name, **model_kwargs) if model_name else None
        self.window_size = window_size
        self.percentile = percentile * 100
        self.device = device
        self.target_chunk_length = target_chunk_length
        self._sentence_transformer_kwargs = sentence_transformer_kwargs

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
            if self.window_size == 0:
                # Each sentence is embedded individually
                windowed_sentences = sentences.copy()
            else:
                # Use the existing windowing logic
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

    def set_model(self, model: Union[str, SENCENCETRANSFORMER_TYPE]) -> None:
        """Set or update the model used for generating embeddings.

        Parameters
        ----------
        model : Union[str, SentenceTransformer]
            Either a string specifying the model name/path or a pre-initialized SentenceTransformer instance
        """
        if isinstance(model, str):
            # Initialize model with additional kwargs if provided
            model_kwargs = {"trust_remote_code": True, "device": self.device}
            if hasattr(self, '_sentence_transformer_kwargs') and self._sentence_transformer_kwargs:
                model_kwargs.update(self._sentence_transformer_kwargs)
            self._model = SentenceTransformer(model, **model_kwargs)
        elif isinstance(model, SentenceTransformer):
            self._model = model
            self.model_name = str(model)[:10]
        else:
            raise TypeError("model must be a str or SentenceTransformer")

    def _create_chunks(self, sentences: List[str], breakpoints: List[int]) -> List[str]:
        """Split sentences into chunks based on breakpoints and optional target length."""
        if not breakpoints:
            return [" ".join(sentences)]

        start = 0
        total_length = 0
        chunk_count = 0

        # First pass: create initial chunks based on semantic breakpoints
        initial_chunks = []
        for bp in breakpoints:
            end = bp + 1
            chunk = " ".join(sentences[start:end])
            initial_chunks.append((chunk, start, end))
            total_length += len(chunk)
            chunk_count += 1
            start = end

        # Add remaining sentences
        if start < len(sentences):
            chunk = " ".join(sentences[start:])
            initial_chunks.append((chunk, start, len(sentences)))
            total_length += len(chunk)
            chunk_count += 1

        if not self.target_chunk_length or chunk_count <= 1:
            return [chunk for chunk, _, _ in initial_chunks]

        # Calculate mean length of current chunks
        mean_length = total_length / chunk_count

        # If mean length is already close to target (within 20%), keep current chunks
        if 0.8 <= mean_length / self.target_chunk_length <= 1.2:
            return [chunk for chunk, _, _ in initial_chunks]

        # Second pass: adjust chunks to better match target length
        final_chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence) + 1  # +1 for space
            
            # If adding this sentence would make the chunk too far from target,
            # start a new chunk unless current chunk is too small
            if (current_length > 0 and 
                abs(current_length - self.target_chunk_length) < 
                abs(current_length + sentence_length - self.target_chunk_length)):
                
                if current_chunk:
                    final_chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            final_chunks.append(" ".join(current_chunk))

        return final_chunks
