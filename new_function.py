def generate_blog_outline(topic):
    """
    Generate a detailed outline (list of section headings) for the given topic.
    Uses YAML format for better readability and flexibility.
    """
    max_attempts = 3
    
    for attempt in range(max_attempts):
        outline_prompt = f"""
Generate a detailed outline for a comprehensive 1500-word **parody article** on "{topic}".
Imagine it as a witty, sarcastic, or over-the-top humorous take.

Format your response as YAML with a 'sections' key containing a list of section headings.
Example:
```yaml
sections:
  - "Ridiculous Introduction: Setting the Stage for Absurdity"
  - "Overblown Claims That Make No Sense"
  - "Expert Opinions from People Who Don't Exist"
  - "Conclusion Full of Regret and Questionable Advice"
```

Include 4-5 creative section headings that would make for an engaging article structure.
"""
        resp = requests.post(
            f"http://{OLLAMA_SERVER}/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": outline_prompt,
                "temperature": 0.4,
                "max_tokens": 1000,  # Increased token count
                "stream": False,
                "device": "cuda",
            },
            timeout=300
        )
        data = resp.json()
        raw_outline = data.get("response", "")
        print(f"🔍 Raw outline response (attempt {attempt+1}):\n{raw_outline}")
        
        # Try to extract YAML content between triple backticks if present
        import re
        yaml_match = re.search(r'```yaml\s*(.*?)\s*```', raw_outline, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
        else:
            yaml_content = raw_outline
            
        try:
            # Parse YAML content
            import yaml
            parsed = yaml.safe_load(yaml_content)
            
            # Check if we have a valid structure with sections
            if isinstance(parsed, dict) and 'sections' in parsed and isinstance(parsed['sections'], list):
                headings = parsed['sections']
                if headings and len(headings) >= 3:  # Ensure we have at least 3 sections
                    return headings
                    
            print(f"⚠️ Attempt {attempt+1}: Invalid YAML structure or insufficient sections")
            
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}: Failed to parse YAML: {e}")
            
        # If we reach here, the attempt failed - continue to next attempt
        if attempt < max_attempts - 1:
            print(f"🔄 Regenerating outline (attempt {attempt+2})...")
    
    # If all attempts fail, make one final attempt with a simpler prompt
    final_prompt = f"""
Create a simple outline for an article about "{topic}".
Format as YAML:

sections:
  - "Introduction"
  - "Main Point 1"
  - "Main Point 2"
  - "Conclusion"
"""
    resp = requests.post(
        f"http://{OLLAMA_SERVER}/api/generate",
        json={
            "model": "llama3:8b",
            "prompt": final_prompt,
            "temperature": 0.2,
            "max_tokens": 500,
            "stream": False,
            "device": "cuda",
        },
        timeout=300
    )
    data = resp.json()
    raw_outline = data.get("response", "")
    
    # Last resort - return basic structure
    return ["Introduction", "Main Content", "Analysis", "Conclusion"]
