#!/usr/bin/env python3
"""
Quick test of the test harness
"""

def test_harness():
    try:
        from test_harness import RAGTestHarness
        print("✅ Test harness imported successfully")
        
        harness = RAGTestHarness()
        print("✅ Test harness initialized")
        
        # Test a single question
        harness.test_question("What is machine learning?")
        
        print("✅ Test harness working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_harness()
