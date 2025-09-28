import json
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BabyRAGSystem:
    def __init__(self, data_path: str = "data/snippets.json", collection_name: str = "tech_snippets"):
        """
        Initialize the Baby RAG System
        
        Args:
            data_path: Path to the JSON file containing text snippets
            collection_name: Name for the Chroma collection
        """
        self.data_path = data_path
        self.collection_name = collection_name

        # Load data
        self.snippets: List[Dict[str, str]] = []
        self._load_data()

        self.collection_size = len(self.snippets)

        # Build vector index with TF-IDF (fast to install)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = None
        self.documents: List[str] = []
        self.vector_dimensions = 0
        self._build_index()
    
    def _load_data(self):
        """Load text snippets from JSON file"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.snippets = json.load(f)
            print(f"Loaded {len(self.snippets)} text snippets")
        except FileNotFoundError:
            print(f"Error: Could not find {self.data_path}")
            raise
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.data_path}")
            raise
    
    def _build_index(self) -> None:
        """Create TF-IDF matrix for the snippets."""
        documents = [f"{snippet['title']}: {snippet['content']}" for snippet in self.snippets]
        self.documents = documents
        print("Generating TF-IDF vectors...")
        self.doc_matrix = self.vectorizer.fit_transform(documents)
        self.vector_dimensions = self.doc_matrix.shape[1]
        print(
            f"Built TF-IDF matrix for {self.doc_matrix.shape[0]} documents "
            f"with {self.vector_dimensions} features"
        )
    
    def query(self, question: str, n_results: int = 3) -> List[Dict]:
        """
        Query the RAG system with a natural language question
        
        Args:
            question: The natural language question
            n_results: Number of results to return
            
        Returns:
            List of relevant snippets with similarity scores
        """
        print(f"\n🔍 Querying: '{question}'")

        if self.doc_matrix is None:
            raise RuntimeError("TF-IDF matrix not initialized")

        query_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vec, self.doc_matrix).flatten()

        limit = max(1, min(n_results, len(self.snippets)))
        top_indices = similarities.argsort()[::-1][:limit]

        formatted_results: List[Dict] = []
        for idx in top_indices:
            snippet = self.snippets[idx]
            similarity = float(similarities[idx])
            formatted_results.append({
                'id': snippet['id'],
                'title': snippet['title'],
                'content': self.documents[idx],
                'similarity_score': similarity,
                'distance': 1 - similarity
            })
        
        return formatted_results
    
    def demo_query(self, question: str):
        """Demo function to show query results in a nice format"""
        results = self.query(question, n_results=3)
        
        print(f"\n📋 Results for: '{question}'")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Similarity Score: {result['similarity_score']:.3f}")
            print(f"   Content: {result['content'][:200]}...")
            print("-" * 30)
        
        return results

    def get_snippets(self) -> List[Dict[str, str]]:
        """Return all snippets with combined title and content."""
        return [
            {
                "id": snippet["id"],
                "title": snippet["title"],
                "content": self.documents[idx],
            }
            for idx, snippet in enumerate(self.snippets)
        ]

def main():
    """Main function to demonstrate the RAG system"""
    print("🚀 Initializing Baby RAG System...")
    
    # Initialize the RAG system
    rag = BabyRAGSystem()
    
    # Demo queries
    demo_questions = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is Python programming?",
        "Tell me about web development",
        "What are databases?",
        "Explain cloud computing",
        "What is cybersecurity?",
        "How does version control work?"
    ]
    
    print("\n" + "="*60)
    print("🎯 DEMO: Baby RAG System in Action")
    print("="*60)
    
    for question in demo_questions:
        rag.demo_query(question)
        input("\nPress Enter to continue to next query...")
    
    # Interactive mode
    print("\n" + "="*60)
    print("💬 Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        user_question = input("\n❓ Your question: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_question:
            rag.demo_query(user_question)

if __name__ == "__main__":
    main()
