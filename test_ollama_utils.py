#!/usr/bin/env python3
"""
Unit tests for the Ollama utilities.
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock
from io import BytesIO

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ollama_utils import (
    is_local_ollama_available,
    get_ollama_endpoint,
    list_available_models,
    generate_text_with_ollama,
    generate_image_with_ollama
)

class TestOllamaUtils(unittest.TestCase):
    """Test cases for the Ollama utilities."""

    @patch('ollama_utils.requests.get')
    def test_is_local_ollama_available_true(self, mock_get):
        """Test is_local_ollama_available when Ollama is available."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Call the function
        result = is_local_ollama_available()

        # Check the result
        self.assertTrue(result)
        mock_get.assert_called_once_with('http://localhost:11434/api/tags', timeout=2)

    @patch('ollama_utils.requests.get')
    def test_is_local_ollama_available_false(self, mock_get):
        """Test is_local_ollama_available when Ollama is not available."""
        # Mock the response to raise an exception
        mock_get.side_effect = Exception("Connection refused")

        # Call the function
        result = is_local_ollama_available()

        # Check the result
        self.assertFalse(result)

    @patch('ollama_utils.is_local_ollama_available')
    def test_get_ollama_endpoint_local(self, mock_is_local):
        """Test get_ollama_endpoint when local Ollama is available."""
        # Mock the response
        mock_is_local.return_value = True

        # Call the function with no environment variable
        with patch.dict('os.environ', {}, clear=True):
            result = get_ollama_endpoint()

        # Check the result
        self.assertEqual(result, 'http://localhost:11434')

    @patch('ollama_utils.is_local_ollama_available')
    def test_get_ollama_endpoint_env(self, mock_is_local):
        """Test get_ollama_endpoint when OLLAMA_SERVER is set."""
        # Mock the response
        mock_is_local.return_value = False

        # Call the function with environment variable
        with patch.dict('os.environ', {'OLLAMA_SERVER': 'example.com:11434'}, clear=True):
            result = get_ollama_endpoint()

        # Check the result
        self.assertEqual(result, 'http://example.com:11434')

    @patch('ollama_utils.requests.get')
    def test_list_available_models(self, mock_get):
        """Test list_available_models."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [
                {'name': 'llama3:8b'},
                {'name': 'sdxl:latest'}
            ]
        }
        mock_get.return_value = mock_response

        # Call the function
        with patch('ollama_utils.get_ollama_endpoint', return_value='http://localhost:11434'):
            result = list_available_models()

        # Check the result
        self.assertEqual(result, ['llama3:8b', 'sdxl:latest'])
        mock_get.assert_called_once_with('http://localhost:11434/api/tags', timeout=5)

    @patch('ollama_utils.requests.post')
    def test_generate_text_with_ollama(self, mock_post):
        """Test generate_text_with_ollama."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Generated text'
        }
        mock_post.return_value = mock_response

        # Call the function
        with patch('ollama_utils.get_ollama_endpoint', return_value='http://localhost:11434'):
            result = generate_text_with_ollama(
                prompt='Test prompt',
                model='llama3:8b',
                temperature=0.7,
                max_tokens=100,
                stream=False,
                device='cuda'
            )

        # Check the result
        self.assertEqual(result, {'response': 'Generated text'})
        mock_post.assert_called_once_with(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3:8b',
                'prompt': 'Test prompt',
                'temperature': 0.7,
                'max_tokens': 100,
                'stream': False,
                'device': 'cuda',
            },
            timeout=300
        )

    @patch('ollama_utils.requests.post')
    def test_generate_image_with_ollama(self, mock_post):
        """Test generate_image_with_ollama."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'image/jpeg'}
        mock_response.content = b'fake image data'
        mock_post.return_value = mock_response

        # Mock list_available_models
        with patch('ollama_utils.list_available_models', return_value=['sdxl:latest']):
            # Call the function
            with patch('ollama_utils.get_ollama_endpoint', return_value='http://localhost:11434'):
                result = generate_image_with_ollama(
                    prompt='Test prompt',
                    model='sdxl:latest',
                    width=800,
                    height=600,
                    steps=30,
                    negative_prompt='blurry'
                )

        # Check the result
        self.assertEqual(result, b'fake image data')
        mock_post.assert_called_once_with(
            'http://localhost:11434/api/generate',
            json={
                'model': 'sdxl:latest',
                'prompt': 'Generate an image: Test prompt. Negative prompt: blurry',
                'format': 'image',
                'options': {
                    'width': 800,
                    'height': 600,
                    'steps': 30,
                }
            },
            timeout=300
        )

if __name__ == '__main__':
    unittest.main()