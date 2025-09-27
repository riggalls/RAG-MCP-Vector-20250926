#!/usr/bin/env python3
"""
Simple test script for the Baby RAG System
"""

def test_rag_system():
    """Test the RAG system with a simple query"""
    try:
        print("🚀 Testing Baby RAG System...")
        
        # Import and initialize
        from rag_system import BabyRAGSystem
        print("✅ Import successful")
        
        # Initialize the system
        rag = BabyRAGSystem()
        print("✅ RAG system initialized")
        
        # Test a simple query
        print("\n🔍 Testing query: 'What is machine learning?'")
        results = rag.query("What is machine learning?", n_results=2)
        
        print(f"✅ Query successful - found {len(results)} results")
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Content: {result['content'][:100]}...")
        
        print("\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_rag_system()
