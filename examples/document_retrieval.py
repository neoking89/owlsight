from owlsight import DocumentSearcher, SentenceTextSplitter

def main():
    # Sample documents on Quantum Mechanics and General Relativity
    docs = {
        "doc1": (
            "Quantum mechanics is a fundamental theory in physics that provides a description of the physical "
            "properties of nature at small scales, typically atomic and subatomic levels. The theory emerged in "
            "the early 20th century through the work of physicists such as Max Planck, Albert Einstein, "
            "Niels Bohr, Werner Heisenberg, and Erwin Schrödinger. It challenges classical mechanics by introducing "
            "the concepts of wave-particle duality, quantization of energy levels, and the probabilistic nature of "
            "physical phenomena. One of the foundational principles is Heisenberg's uncertainty principle, which states "
            "that the position and momentum of a particle cannot both be precisely determined at the same time. Another "
            "core concept is quantum entanglement, where two or more particles become correlated such that the measurement "
            "of one instantaneously influences the state of the other, regardless of distance, a phenomenon Einstein called "
            "'spooky action at a distance'. Despite its counterintuitive implications, quantum mechanics forms the basis "
            "of modern technologies such as semiconductors, lasers, and quantum computing."
        ),
        "doc2": (
            "General relativity, formulated by Albert Einstein in 1915, is a geometric theory of gravitation that "
            "revolutionized our understanding of space and time. It describes gravity not as a force in the Newtonian sense, "
            "but as the curvature of spacetime caused by mass and energy. The Einstein field equations (EFE) relate this curvature "
            "to the energy-momentum tensor, which encapsulates the distribution of matter and energy in spacetime. One of the "
            "most famous predictions of general relativity is the existence of black holes, regions of spacetime where gravity "
            "is so strong that nothing, not even light, can escape. Another remarkable prediction is gravitational waves—ripples "
            "in the fabric of spacetime caused by accelerating massive objects, confirmed observationally by LIGO in 2015. "
            "General relativity is also crucial for understanding cosmology, including the expansion of the universe, the Big Bang, "
            "and dark energy. While it has been extensively tested in weak-field conditions, it remains incompatible with quantum "
            "mechanics, sparking the search for a theory of quantum gravity."
        ),
        "doc3": (
            "The incompatibility between quantum mechanics and general relativity has led to the development of theories "
            "aimed at unifying the two frameworks. String theory proposes that fundamental particles are not point-like but "
            "rather one-dimensional strings whose vibrations determine their properties. Another approach is loop quantum gravity "
            "(LQG), which attempts to quantize spacetime itself, suggesting that space is composed of discrete, quantized units. "
            "Efforts to reconcile quantum mechanics and general relativity also explore concepts like holography, which posits "
            "that the universe may be described by lower-dimensional boundary theories. A key challenge in quantum gravity is "
            "understanding how the fabric of spacetime emerges from quantum interactions. Experimental evidence remains elusive, "
            "but studies of black hole thermodynamics, the holographic principle, and potential observable effects in the early "
            "universe continue to provide insights. The resolution of this conflict between the two pillars of modern physics "
            "may fundamentally alter our understanding of reality and the nature of the universe."
        )
    }

    splitter = SentenceTextSplitter(n_sentences=4, n_overlap=1)
    searcher = DocumentSearcher(
        documents=docs,
        text_splitter=splitter,
        cache_dir="document_cache",
        cache_dir_suffix="physics"
    )
    
    query = "quantum gravity and black holes"
    results = searcher.search(query, top_k=5)
    
    print(f"\nSearch Results for '{query}':")
    print("=" * 80)
    for _, row in results.iterrows():
        print(f"\nDocument: {row['document_name']}")
        print(f"Score: {row['aggregated_score']:.4f}")
        print(f"Content: {row['document']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
