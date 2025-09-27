import json
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import os

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
        
        # Initialize the embedding model
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        self.client = chromadb.Client()
        
        # Load data and create collection
        self.collection = None
        self.snippets = []
        self._load_data()
        self._create_collection()
    
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
    
    def _create_collection(self):
        """Create Chroma collection and add embeddings"""
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(name=self.collection_name)
            print(f"Created new collection: {self.collection_name}")
        
        # Check if collection is empty
        if self.collection.count() == 0:
            print("Generating embeddings and adding to collection...")
            self._add_snippets_to_collection()
        else:
            print(f"Collection already contains {self.collection.count()} documents")
    
    def _add_snippets_to_collection(self):
        """Generate embeddings and add snippets to Chroma collection"""
        documents = []
        metadatas = []
        ids = []
        
        for snippet in self.snippets:
            # Combine title and content for better context
            full_text = f"{snippet['title']}: {snippet['content']}"
            documents.append(full_text)
            metadatas.append({
                "title": snippet['title'],
                "id": snippet['id']
            })
            ids.append(str(snippet['id']))
        
        # Add to collection (Chroma will generate embeddings automatically)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} documents to collection")
    
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
        
        # Query the collection
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            doc_id = results['ids'][0][i]
            document = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Convert distance to similarity score (higher is better)
            similarity_score = 1 - distance
            
            formatted_results.append({
                'id': doc_id,
                'title': metadata['title'],
                'content': document,
                'similarity_score': similarity_score,
                'distance': distance
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
