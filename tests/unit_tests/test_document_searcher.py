import pytest

from owlsight.rag.core import DocumentSearcher


def test_split_documents_basic():
    """Test basic functionality of split_documents with default parameters."""
    documents = {
        "doc1": "This is sentence one. This is sentence two. This is sentence three. This is sentence four.",
        "doc2": "Short doc. With few. Sentences only.",
    }
    
    result = DocumentSearcher.split_documents(documents, n_sentences=3, n_overlap=1)
    
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
    result = DocumentSearcher.split_documents(
        documents,
        n_sentences=4,
        n_overlap=2
    )
    
    print("\nTest split_documents_custom_params:")
    print(f"Input document: {documents['doc1']}")
    print(f"Result: {result}")
    
    assert len(result) == 4
    for i in range(3):
        key = f"doc1__split{i}"
        print(f"Checking chunk {i}: {result.get(key, 'NOT FOUND')}")
        assert key in result
    
    assert result["doc1__split0"] == "One. Two. Three. Four."
    assert result["doc1__split1"] == "Three. Four. Five. Six."
    assert result["doc1__split2"] == "Five. Six. Seven. Eight."


def test_split_documents_edge_cases():
    """Test split_documents with edge cases."""
    documents = {
        "empty": "",
        "single": "Just one sentence.",
        "no_periods": "This is a sentence without proper punctuation",
    }
    
    result = DocumentSearcher.split_documents(documents, n_sentences=2)
    
    # Empty document should create empty split
    assert "empty__split0" in result
    assert result["empty__split0"] == ""
    
    # Single sentence should be in one chunk
    assert "single__split0" in result
    assert result["single__split0"] == "Just one sentence."
    
    # Sentence without period should be treated as one sentence
    assert "no_periods__split0" in result
    assert result["no_periods__split0"] == "This is a sentence without proper punctuation"


def test_split_documents_validation():
    """Test input validation in split_documents."""
    documents = {"doc": "Some text."}
    
    # Test n_overlap >= n_sentences
    with pytest.raises(ValueError):
        DocumentSearcher.split_documents(documents, n_sentences=2, n_overlap=2)
    
    with pytest.raises(ValueError):
        DocumentSearcher.split_documents(documents, n_sentences=2, n_overlap=3)
