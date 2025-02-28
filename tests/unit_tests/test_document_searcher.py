import pytest

from owlsight.rag.core import DocumentSearcher, SearchMethod, SentenceTextSplitter
from unittest.mock import patch, MagicMock


def test_split_documents_basic():
    """Test basic functionality of split_documents with default parameters."""
    documents = {
        "doc1": "This is sentence one. This is sentence two. This is sentence three. This is sentence four.",
        "doc2": "Short doc. With few. Sentences only.",
    }

    splitter = SentenceTextSplitter(n_sentences=3, n_overlap=1)
    result = splitter.split_documents(documents)

    # Check doc1 splits (should have 2 chunks with 1 sentence overlap)
    assert "doc1__split0" in result
    assert "doc1__split1" in result
    assert result["doc1__split0"] == "This is sentence one. This is sentence two. This is sentence three."
    assert result["doc1__split1"] == "This is sentence three. This is sentence four."

    # Check doc2 (should be one chunk as it's only 3 sentences)
    assert "doc2__split0" in result
    assert result["doc2__split0"] == "Short doc. With few. Sentences only."


def test_split_documents_custom_params():
    """Test split_documents with custom n_sentences and n_overlap."""
    documents = {
        "doc1": "One. Two. Three. Four. Five. Six. Seven. Eight.",
    }

    # Split into chunks of 4 sentences with 2 sentence overlap
    splitter = SentenceTextSplitter(n_sentences=4, n_overlap=2)
    result = splitter.split_documents(documents)

    print("\nTest split_documents_custom_params:")
    print(f"Input document: {documents['doc1']}")
    print("\nActual result chunks:")
    for key, value in sorted(result.items()):
        print(f"{key}: {value}")

    expected_chunks = {
        "doc1__split0": "One. Two. Three. Four.",
        "doc1__split1": "Three. Four. Five. Six.",
        "doc1__split2": "Five. Six. Seven. Eight.",
    }
    print("\nExpected chunks:")
    for key, value in sorted(expected_chunks.items()):
        print(f"{key}: {value}")

    print("\nSplitter settings:")
    print(f"n_sentences: {splitter.n_sentences}")
    print(f"n_overlap: {splitter.n_overlap}")

    assert result == expected_chunks


def test_split_documents_edge_cases():
    """Test split_documents with edge cases."""
    documents = {
        "empty": "",
        "single": "Just one sentence.",
        "no_periods": "This is a sentence without proper punctuation",
    }

    splitter = SentenceTextSplitter(n_sentences=2)
    result = splitter.split_documents(documents)

    # Empty document should create empty split
    assert "empty__split0" in result
    assert result["empty__split0"] == ""

    # Single sentence should be in one chunk
    assert "single__split0" in result
    assert result["single__split0"] == "Just one sentence."

    # Sentence without period should be treated as one sentence
    assert "no_periods__split0" in result
    assert result["no_periods__split0"] == "This is a sentence without proper punctuation"


def test_sentence_text_splitter_validation():
    """Test input validation in SentenceTextSplitter constructor."""
    # Test n_overlap >= n_sentences
    with pytest.raises(ValueError):
        SentenceTextSplitter(n_sentences=2, n_overlap=2)

    with pytest.raises(ValueError):
        SentenceTextSplitter(n_sentences=2, n_overlap=3)


def test_document_searcher_init_basic():
    """Test the basic initialization of DocumentSearcher."""
    documents = {"doc1": "Python is a programming language", "doc2": "Machine learning is fascinating"}
    splitter = SentenceTextSplitter(n_sentences=3, n_overlap=1)
    expected_documents = splitter.split_documents(documents)

    with patch("owlsight.rag.core.EnsembleSearchEngine", new_callable=MagicMock) as mock_engine:
        searcher = DocumentSearcher(documents, text_splitter=splitter)

        # Check if the EnsembleSearchEngine was called with the expected arguments
        mock_engine.assert_called_once_with(
            documents=expected_documents,
            search_methods=[SearchMethod.TFIDF, SearchMethod.SENTENCE_TRANSFORMER],
            cache_dir=None,
            cache_dir_suffix=None,
            init_arguments={
                SearchMethod.SENTENCE_TRANSFORMER: {
                    "pooling_strategy": "mean",
                    "model_name": searcher.sentence_transformer_model,
                    "batch_size": searcher.sentence_transformer_batch_size,
                    "device": None,
                }
            },
        )


def test_document_searcher_init_with_cache():
    """Test initialization of DocumentSearcher with cache directory and suffix."""
    documents = {"doc1": "Python is a programming language", "doc2": "Machine learning is fascinating"}
    cache_dir = "test_cache"
    base_cache_suffix = "test_suffix"
    splitter = SentenceTextSplitter(n_sentences=3, n_overlap=1)
    expected_cache_suffix = f"{base_cache_suffix}__SentenceTextSplitter__n_sentences=3__n_overlap=1"

    with patch("owlsight.rag.core.EnsembleSearchEngine", new_callable=MagicMock) as mock_engine:
        searcher = DocumentSearcher(
            documents, cache_dir=cache_dir, cache_dir_suffix=base_cache_suffix, text_splitter=splitter
        )

        # Check if the EnsembleSearchEngine was called with the expected arguments
        mock_engine.assert_called_once_with(
            documents=searcher.documents,
            search_methods=[SearchMethod.TFIDF, SearchMethod.SENTENCE_TRANSFORMER],
            cache_dir=cache_dir,
            cache_dir_suffix=expected_cache_suffix,
            init_arguments={
                SearchMethod.SENTENCE_TRANSFORMER: {
                    "pooling_strategy": "mean",
                    "model_name": searcher.sentence_transformer_model,
                    "batch_size": searcher.sentence_transformer_batch_size,
                    "device": None,
                }
            },
        )
