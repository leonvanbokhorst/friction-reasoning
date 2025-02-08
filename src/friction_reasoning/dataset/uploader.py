"""HuggingFace Hub dataset upload functionality."""

from pathlib import Path
from typing import Optional, Dict, Any
import json

from datasets import Dataset
from huggingface_hub import HfApi

class HuggingFaceUploader:
    """Handles uploading datasets to HuggingFace Hub."""
    
    def __init__(self, repo_id: str, token: Optional[str] = None):
        """Initialize the uploader.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
            token: HuggingFace API token (default: None, will use env var)
        """
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        
    def upload_jsonl(self, file_path: str, split: str = "train") -> str:
        """Upload a JSONL file as a dataset split.
        
        Args:
            file_path: Path to the JSONL file
            split: Dataset split name (default: "train")
            
        Returns:
            URL of the uploaded dataset on HuggingFace Hub
        """
        # Load JSONL file
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))
                
        # Create dataset
        dataset = Dataset.from_list(examples)
        
        # Push to hub
        url = dataset.push_to_hub(
            self.repo_id,
            split=split,
            private=True  # Change to False if dataset should be public
        )
        
        return url
    
    def update_metadata(self, metadata: Dict[str, Any]):
        """Update dataset metadata on HuggingFace Hub.
        
        Args:
            metadata: Dictionary of metadata to update
        """
        # Only pass valid parameters to update_repo_settings
        repo_settings = {
            "private": metadata.get("private", True),
        }
        
        # Update the repository settings
        self.api.update_repo_settings(
            repo_id=self.repo_id,
            **repo_settings
        )
        
        # Update the dataset card with full metadata
        self.api.upload_file(
            repo_id=self.repo_id,
            path_or_fileobj=json.dumps(metadata, indent=2).encode(),
            path_in_repo="dataset_info.json"
        )