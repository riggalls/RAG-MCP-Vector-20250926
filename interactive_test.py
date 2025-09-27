#!/usr/bin/env python3
"""
Interactive test script for the Baby RAG System
"""

from rag_system import BabyRAGSystem

def main():
    print("🚀 Baby RAG System - Interactive Test")
    print("=" * 50)
    
    # Initialize the system
    print("Initializing RAG system...")
    rag = BabyRAGSystem()
    print("✅ Ready to answer questions!")
    
    # Test queries
    test_questions = [
        "What is machine learning?",
        "How do neural networks work?", 
        "Tell me about Python programming",
        "What are databases?",
        "Explain cloud computing"
    ]
    
    print(f"\n📋 Testing {len(test_questions)} sample questions:")
    print("-" * 30)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing: '{question}'")
        results = rag.query(question, n_results=1)
        
        if results:
            best_result = results[0]
            print(f"   ✅ Best match: {best_result['title']}")
            print(f"   📊 Similarity: {best_result['similarity_score']:.3f}")
            print(f"   📝 Content: {best_result['content'][:150]}...")
        else:
            print("   ❌ No results found")
    
    print("\n" + "=" * 50)
    print("🎯 Interactive Mode - Ask your own questions!")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            user_question = input("\n❓ Your question: ").strip()
            
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_question:
                print(f"\n🔍 Searching for: '{user_question}'")
                results = rag.query(user_question, n_results=3)
                
                if results:
                    print(f"\n📋 Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. {result['title']}")
                        print(f"   📊 Similarity: {result['similarity_score']:.3f}")
                        print(f"   📝 Content: {result['content'][:200]}...")
                        print("-" * 40)
                else:
                    print("❌ No relevant results found")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
