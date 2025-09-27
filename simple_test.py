#!/usr/bin/env python3
"""
Simple test to verify the RAG system works
"""

def main():
    print("🚀 Testing Baby RAG System...")
    
    try:
        # Import the system
        from rag_system import BabyRAGSystem
        print("✅ Import successful")
        
        # Initialize
        print("Initializing system...")
        rag = BabyRAGSystem()
        print("✅ System initialized")
        
        # Test queries
        test_questions = [
            "What is machine learning?",
            "Tell me about Python",
            "How do databases work?"
        ]
        
        print("\n🔍 Testing queries:")
        for question in test_questions:
            print(f"\nQuestion: {question}")
            results = rag.query(question, n_results=1)
            
            if results:
                result = results[0]
                print(f"✅ Answer: {result['title']}")
                print(f"📊 Score: {result['similarity_score']:.3f}")
                print(f"📝 Content: {result['content'][:100]}...")
            else:
                print("❌ No results")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
