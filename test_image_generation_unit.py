#!/usr/bin/env python3
"""
Unit tests for the enhanced image generation functionality.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions to test
from post import generate_image
import apple_utils

class TestImageGeneration(unittest.TestCase):
    """Test the enhanced image generation functionality."""
    
    @patch('apple_utils.is_apple_silicon')
    @patch('apple_utils.generate_image_with_mlx')
    @patch('requests.post')
    def test_generate_image_apple_silicon(self, mock_post, mock_generate_mlx, mock_is_apple):
        """Test image generation on Apple Silicon."""
        # Mock Apple Silicon detection to return True
        mock_is_apple.return_value = True
        
        # Mock successful MLX generation
        mock_generate_mlx.return_value = "/tmp/test_image.webp"
        
        # Call generate_image
        result = generate_image("test prompt")
        
        # Verify MLX was used and AUTOMATIC1111 was not called
        mock_generate_mlx.assert_called_once()
        mock_post.assert_not_called()
        
        # Check the result
        self.assertEqual(result, "/tmp/test_image.webp")
        print("✅ Test passed: generate_image correctly uses MLX Core on Apple Silicon")
    
    @patch('apple_utils.is_apple_silicon')
    @patch('requests.post')
    def test_generate_image_non_apple(self, mock_post, mock_is_apple):
        """Test image generation on non-Apple hardware."""
        # Mock Apple Silicon detection to return False
        mock_is_apple.return_value = False
        
        # Mock AUTOMATIC1111 API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"images": ["base64_encoded_image_data"]}
        mock_post.return_value = mock_response
        
        # Mock PIL and file operations
        with patch('PIL.Image.open') as mock_open, \
             patch('builtins.open', create=True), \
             patch('os.path.exists') as mock_exists, \
             patch('PIL.Image.Image.save'):
            
            # Mock that the cache file doesn't exist
            mock_exists.return_value = False
            
            # Mock the image object
            mock_img = MagicMock()
            mock_open.return_value = mock_img
            
            # Call generate_image
            result = generate_image("test prompt")
            
            # Verify AUTOMATIC1111 was called
            mock_post.assert_called_once()
            
            # Check that the result is not None
            self.assertIsNotNone(result)
            print("✅ Test passed: generate_image correctly uses AUTOMATIC1111 on non-Apple hardware")
    
    @patch('apple_utils.is_apple_silicon')
    @patch('apple_utils.generate_image_with_mlx')
    @patch('requests.post')
    def test_generate_image_fallback(self, mock_post, mock_generate_mlx, mock_is_apple):
        """Test fallback to AUTOMATIC1111 when MLX fails."""
        # Mock Apple Silicon detection to return True
        mock_is_apple.return_value = True
        
        # Mock MLX generation failure
        mock_generate_mlx.return_value = None
        
        # Mock AUTOMATIC1111 API response
        mock_response = MagicMock()
        mock_response.json.return_value = {"images": ["base64_encoded_image_data"]}
        mock_post.return_value = mock_response
        
        # Mock PIL and file operations
        with patch('PIL.Image.open') as mock_open, \
             patch('builtins.open', create=True), \
             patch('os.path.exists') as mock_exists, \
             patch('PIL.Image.Image.save'):
            
            # Mock that the cache file doesn't exist
            mock_exists.return_value = False
            
            # Mock the image object
            mock_img = MagicMock()
            mock_open.return_value = mock_img
            
            # Call generate_image
            result = generate_image("test prompt")
            
            # Verify both MLX and AUTOMATIC1111 were called
            mock_generate_mlx.assert_called_once()
            mock_post.assert_called_once()
            
            # Check that the result is not None
            self.assertIsNotNone(result)
            print("✅ Test passed: generate_image correctly falls back to AUTOMATIC1111 when MLX fails")

if __name__ == "__main__":
    unittest.main()