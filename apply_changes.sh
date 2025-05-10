#!/bin/bash

# Backup the original file
cp post.py post.py.bak

# Replace the original file with the new one
cp post.py.new post.py

echo "Changes applied successfully. Original file backed up as post.py.bak"