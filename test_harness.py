#!/usr/bin/env python3
"""
Simple Test Harness for Baby RAG System
Pass in your questions and get results!
"""

import sys
import json
from rag_system import BabyRAGSystem

class RAGTestHarness:
    def __init__(self):
        """Initialize the test harness"""
        print("🚀 Initializing RAG Test Harness...")
        self.rag = BabyRAGSystem()
        print("✅ Ready to test your questions!")
    
    def test_question(self, question, n_results=3):
        """Test a single question"""
        print(f"\n🔍 Testing: '{question}'")
        print("-" * 50)
        
        try:
            results = self.rag.query(question, n_results=n_results)
            
            if results:
                print(f"📋 Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['title']}")
                    print(f"   📊 Similarity Score: {result['similarity_score']:.3f}")
                    print(f"   📝 Content: {result['content']}")
                    print("-" * 30)
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def test_questions_from_list(self, questions):
        """Test multiple questions from a list"""
        print(f"\n📋 Testing {len(questions)} questions:")
        print("=" * 60)
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {question}")
            self.test_question(question, n_results=1)
    
    def test_questions_from_file(self, filename):
        """Test questions from a JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and 'questions' in data:
                questions = data['questions']
            else:
                print("❌ Invalid file format. Expected list of questions or dict with 'questions' key")
                return
            
            self.test_questions_from_list(questions)
            
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in file: {filename}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    
    def interactive_mode(self):
        """Interactive mode for testing questions"""
        print("\n" + "=" * 60)
        print("💬 Interactive Test Mode")
        print("Type your questions, 'quit' to exit, 'help' for commands")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n❓ Enter question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    print("\n📖 Available commands:")
                    print("  - Type any question to test it")
                    print("  - 'quit' or 'exit' to stop")
                    print("  - 'help' to show this message")
                    continue
                elif user_input:
                    self.test_question(user_input)
                else:
                    print("Please enter a question or 'quit' to exit")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function with command line interface"""
    harness = RAGTestHarness()
    
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == "--interactive" or sys.argv[1] == "-i":
            harness.interactive_mode()
        elif sys.argv[1] == "--file" or sys.argv[1] == "-f":
            if len(sys.argv) > 2:
                harness.test_questions_from_file(sys.argv[2])
            else:
                print("❌ Please provide a filename: python test_harness.py --file questions.json")
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("\n📖 RAG Test Harness Usage:")
            print("  python test_harness.py                    # Interactive mode")
            print("  python test_harness.py --interactive      # Interactive mode")
            print("  python test_harness.py --file questions.json  # Test from file")
            print("  python test_harness.py --help             # Show this help")
        else:
            # Single question mode
            question = " ".join(sys.argv[1:])
            harness.test_question(question)
    else:
        # Default to interactive mode
        harness.interactive_mode()

if __name__ == "__main__":
    main()
