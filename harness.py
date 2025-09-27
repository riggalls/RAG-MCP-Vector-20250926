#!/usr/bin/env python3
"""
Simple RAG Test Harness - Direct Question Testing
"""

import sys

def test_single_question(question):
    """Test a single question directly"""
    print(f"🚀 Testing question: '{question}'")
    print("Loading RAG system...")
    
    try:
        from rag_system import BabyRAGSystem
        rag = BabyRAGSystem()
        
        print(f"\n🔍 Querying: '{question}'")
        results = rag.query(question, n_results=3)
        
        if results:
            print(f"\n✅ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']}")
                print(f"   Score: {result['similarity_score']:.3f}")
                print(f"   Text: {result['content'][:150]}...")
        else:
            print("❌ No results found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_multiple_questions(questions):
    """Test multiple questions"""
    print(f"🚀 Testing {len(questions)} questions...")
    print("Loading RAG system...")
    
    try:
        from rag_system import BabyRAGSystem
        rag = BabyRAGSystem()
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Testing: '{question}'")
            results = rag.query(question, n_results=1)
            
            if results:
                best = results[0]
                print(f"   ✅ Best match: {best['title']} (Score: {best['similarity_score']:.3f})")
            else:
                print("   ❌ No match found")
                
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python harness.py 'your question here'")
        print("  python harness.py --batch")
        return
    
    if sys.argv[1] == '--batch':
        # Test predefined questions
        questions = [
            "What is machine learning?",
            "How do neural networks work?",
            "Tell me about Python",
            "What are databases?",
            "Explain cloud computing"
        ]
        test_multiple_questions(questions)
    else:
        # Test single question
        question = " ".join(sys.argv[1:])
        test_single_question(question)

if __name__ == "__main__":
    main()
