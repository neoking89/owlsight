import unittest

from owlsight.rag.core import DocumentSearcher


class TestDocumentSearcher(unittest.TestCase):
    def test_split_documents_basic(self):
        """Test basic functionality of split_documents with default parameters."""
        documents = {
            "doc1": "This is sentence one. This is sentence two. This is sentence three. This is sentence four.",
            "doc2": "Short doc. With few. Sentences only.",
        }
        
        result = DocumentSearcher.split_documents(documents, n_sentences=3, n_overlap=1)
        
        # Check doc1 splits (should have 2 chunks with 1 sentence overlap)
        self.assertIn("doc1__split0", result)
        self.assertIn("doc1__split1", result)
        self.assertEqual(
            result["doc1__split0"],
            "This is sentence one. This is sentence two. This is sentence three."
        )
        self.assertEqual(
            result["doc1__split1"],
            "This is sentence three. This is sentence four."
        )
        
        # Check doc2 (should be one chunk as it's only 3 sentences)
        self.assertIn("doc2__split0", result)
        self.assertEqual(
            result["doc2__split0"],
            "Short doc. With few. Sentences only."
        )

    def test_split_documents_custom_params(self):
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
        
        self.assertEqual(len(result), 3)  # Should have 3 chunks
        for i in range(3):
            key = f"doc1__split{i}"
            print(f"Checking chunk {i}: {result.get(key, 'NOT FOUND')}")
            self.assertIn(key, result)
        
        self.assertEqual(
            result["doc1__split0"],
            "One. Two. Three. Four."
        )
        self.assertEqual(
            result["doc1__split1"],
            "Three. Four. Five. Six."
        )
        self.assertEqual(
            result["doc1__split2"],
            "Five. Six. Seven. Eight."
        )

    def test_split_documents_edge_cases(self):
        """Test split_documents with edge cases."""
        documents = {
            "empty": "",
            "single": "Just one sentence.",
            "no_periods": "This is a sentence without proper punctuation",
        }
        
        result = DocumentSearcher.split_documents(documents, n_sentences=2)
        
        # Empty document should create empty split
        self.assertIn("empty__split0", result)
        self.assertEqual(result["empty__split0"], "")
        
        # Single sentence should be in one chunk
        self.assertIn("single__split0", result)
        self.assertEqual(result["single__split0"], "Just one sentence.")
        
        # Sentence without period should be treated as one sentence
        self.assertIn("no_periods__split0", result)
        self.assertEqual(
            result["no_periods__split0"],
            "This is a sentence without proper punctuation"
        )

    def test_split_documents_validation(self):
        """Test input validation in split_documents."""
        documents = {"doc": "Some text."}
        
        # Test n_overlap >= n_sentences
        with self.assertRaises(ValueError):
            DocumentSearcher.split_documents(documents, n_sentences=2, n_overlap=2)
        
        with self.assertRaises(ValueError):
            DocumentSearcher.split_documents(documents, n_sentences=2, n_overlap=3)


if __name__ == "__main__":
    unittest.main()
