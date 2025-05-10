# Changes to Eliminate Redundant LLM Calls

## Problem

The original code had two separate functions that made redundant LLM calls:

1. `generate_blog_components(topic)` - Made an LLM call to generate metadata
2. `generate_blog_outline(topic)` - Made another LLM call to generate section headings

This was inefficient because both calls could be combined into a single LLM call.

## Solution

The solution combines these two LLM calls into a single call by:

1. Modifying the metadata prompt in `generate_blog_components` to request both metadata and section headings
2. Adding a `sections` field to the YAML template with example values
3. Increasing the token limit from 300 to 500 to accommodate the additional content
4. Adding code to extract section headings from the metadata response
5. Removing the call to `generate_blog_outline`
6. Marking `generate_blog_outline` as deprecated but keeping it for backward compatibility

## Files Changed

1. `post.py` - Modified to implement the solution

## How to Apply the Changes

1. Backup the original file:
   ```bash
   cp post.py post.py.bak
   ```

2. Apply the changes:
   ```bash
   cp post.py.new post.py
   ```

3. Test the changes:
   ```bash
   python test_changes.py
   ```

## Benefits

1. Reduced latency - One less LLM call means faster blog generation
2. Lower costs - Fewer API calls to the LLM service
3. Simplified code - The logic is more streamlined and easier to understand

## Potential Issues

1. If the LLM fails to generate valid YAML with the `sections` field, the code falls back to default section headings
2. The increased token limit (from 300 to 500) might slightly increase the cost per call, but this is offset by eliminating the second call entirely