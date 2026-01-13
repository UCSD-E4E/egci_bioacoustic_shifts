"""
SAM-Audio Preprocessor for PyTorch pipelines
Uses Docker Service to avoid dependency conflicts
"""

import torch
import numpy as np
import requests
from typing import Optional, Union

class SamAudioPreprocessor:
  """
  Preprocessor that applies SAM-Audio separation via Docker API

  Usage:
    # Initialize once
    sam_preprocessor = SAMAudioPreprocessor(description="A bird vocalizing")

    # Use in pipeline
    ads.set_transform(sam_preprocessor)
  """

  def __init__(
      self,
      description: str = "A bird vocalizing",
      api_url: str = "http://localhost:5000"
  ):
    """
    Args:
      description: What sound to isolate
      api_url: URL of SAM-Audio Docker Service
    """
    self.description = description
    self.api_url = api_url
    self.separate_endpoint = f"{api_url}/separate"
    self.health_endpoint = f"{api_url}/health"

    self._check_service()
  
  def _check_service(self):
    """Check if Docker Service is accessible"""

    try:
      response = requests.get(self.health_endpoint, timeout = 5)

      if response.status_code == 200:
        print("✓ SAM-Audio service connected")
      else:
        print(f"⚠ SAM-Audio service returned status {response.status_code}")
    except:
      