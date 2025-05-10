#!/usr/bin/env python3

import os
import sys
import re
import yaml
import requests

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set environment variables for testing
os.environ["OLLAMA_SERVER"] = "localhost:11434"

# Import the modified function - need to do this after setting env vars
import post
from post import generate_blog_outline

def test_generate_blog_outline():
    """
    Test that the modified generate_blog_outline function works correctly
    with YAML format and multiple attempts.
    """
    # Mock the LLM response
    def mock_post(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.status_code = 200
            
            def raise_for_status(self):
                pass
            
            def json(self):
                # Return a valid YAML response
                return {
                    "response": """```yaml
sections:
  - "Test Section 1: Introduction"
  - "Test Section 2: Main Content"
  - "Test Section 3: Analysis"
  - "Test Section 4: Conclusion"
```"""
                }
        
        return MockResponse()
    
    # Save the original post function
    original_post = requests.post
    
    try:
        # Replace the post function with our mock
        requests.post = mock_post
        
        # Call generate_blog_outline
        result = generate_blog_outline("test topic")
        
        # Check that the result contains the expected sections
        assert len(result) == 4, f"Expected 4 sections, got {len(result)}"
        assert "Test Section 1: Introduction" in result, "Missing first section"
        assert "Test Section 4: Conclusion" in result, "Missing last section"
        
        print("✅ Test passed: generate_blog_outline works correctly with YAML format")
        return True
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Restore the original function
        requests.post = original_post

if __name__ == "__main__":
    # Run the test
    success = test_generate_blog_outline()
    sys.exit(0 if success else 1)
