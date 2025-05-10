#!/usr/bin/env python3

import re

# Read the original file
with open('post.py', 'r') as f:
    original_content = f.read()

# Read the modified function
with open('modified_post.py', 'r') as f:
    modified_function = f.read()

# Define the pattern to match the original generate_blog_components function
pattern = r'def generate_blog_components\(topic\):.*?return full_raw'
# Use DOTALL flag to match across multiple lines
original_function_match = re.search(pattern, original_content, re.DOTALL)

if original_function_match:
    # Replace the original function with the modified one
    updated_content = original_content.replace(
        original_function_match.group(0),
        modified_function.strip()
    )
    
    # Write the updated content back to the file
    with open('post.py', 'w') as f:
        f.write(updated_content)
    
    print("Successfully updated post.py with the modified generate_blog_components function.")
else:
    print("Could not find the generate_blog_components function in post.py.")