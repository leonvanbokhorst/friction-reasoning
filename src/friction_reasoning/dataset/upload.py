"""Functions for uploading datasets to Hugging Face Hub."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import HfApi

# Load environment variables
load_dotenv()

def create_dataset_card(
    dataset_name: str,
    description: str,
    num_examples: int,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Create a dataset card markdown for the Hugging Face Hub.
    
    Args:
        dataset_name: Name of the dataset
        description: Description of the dataset
        num_examples: Number of examples in the dataset
        metadata: Optional additional metadata
        
    Returns:
        Markdown string for the dataset card
    """
    metadata = metadata or {}
    
    return f"""---
annotations_creators:
- machine-generated
language_creators:
- machine-generated
language:
- en
license:
- mit
multilinguality:
- monolingual
pretty_name: {dataset_name}
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- conversational
- question-answering
task_ids:
- open-domain-qa
---

# Dataset Card for {dataset_name}

## Dataset Description

- **Repository:** {metadata.get('repo_id', 'Not specified')}
- **Size:** {num_examples} examples
- **Version:** {metadata.get('version', '1.0.0')}
- **License:** MIT

### Dataset Summary

{description}

### Data Fields

- `id`: Unique identifier for the example
- `question`: The input question or thought prompt
- `agent_responses`: List of agent reasoning processes
  - `agent_type`: Type of reasoning agent
  - `thought_stream`: Raw stream-of-consciousness thoughts
  - `friction_moments`: Identified moments of cognitive friction
- `final_answer`: Synthesized response incorporating all agent perspectives
- `metadata`: Additional information about the example
  - `timestamp`: When the example was generated
  - `model`: Model used for generation

### Data Splits

- Training: {num_examples} examples
"""

async def upload_to_hub(
    dataset_path: str,
    repo_id: str = "leonvanbokhorst/friction-rambling-v1",
    description: str = "A dataset of multi-agent reasoning with designed friction points.",
    private: bool = False
) -> None:
    """Upload a dataset to Hugging Face Hub.
    
    Args:
        dataset_path: Path to the JSONL dataset file
        repo_id: Hugging Face Hub repository ID
        description: Dataset description for the dataset card
        private: Whether to make the repository private
        
    Raises:
        ValueError: If HUGGINGFACE_API_KEY is not found
        Exception: If upload fails
    """
    # Load environment variables from .env file
    env_path = Path(__file__).parents[3] / ".env"
    if not env_path.exists():
        raise ValueError(f".env file not found at {env_path}")
    load_dotenv(env_path)
    
    # Get API token from environment
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
    
    print(f"\nUploading dataset to Hugging Face Hub: {repo_id}")
    print(f"Using token: {hf_token[:8]}...")  # Print first 8 chars for verification
    
    try:
        # Load the dataset
        print("\nLoading dataset from file...")
        dataset = Dataset.from_json(dataset_path)
        num_examples = len(dataset)
        print(f"Loaded {num_examples} examples")
        
        # Create dataset card
        print("\nCreating dataset card...")
        metadata = {
            "repo_id": repo_id,
            "version": "1.0.0",
            "num_examples": num_examples
        }
        
        dataset_name = repo_id.split("/")[-1]
        dataset_card = create_dataset_card(
            dataset_name=dataset_name,
            description=description,
            num_examples=num_examples,
            metadata=metadata
        )
        
        # Initialize the Hugging Face API
        print("\nInitializing Hugging Face API...")
        api = HfApi(token=hf_token)
        
        # Create the repository if it doesn't exist
        print("\nCreating/verifying repository...")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
            print(f"Repository {repo_id} created/verified.")
        except Exception as e:
            print(f"Error: Could not create/verify repository: {e}")
            raise
            
        # Upload the README.md
        print("\nUploading dataset card...")
        try:
            api.upload_file(
                path_or_fileobj=dataset_card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                create_pr=False,
            )
            print("Dataset card uploaded successfully.")
        except Exception as e:
            print(f"Error: Could not update README.md: {e}")
            raise
        
        # Push dataset to hub
        print("\nPushing dataset to hub (this may take a while)...")
        try:
            dataset.push_to_hub(
                repo_id,
                token=hf_token,
                private=private,
                commit_message="Update friction reasoning dataset",
                embed_external_files=False
            )
            print(f"\nDataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"Error: Failed to push dataset to hub: {e}")
            raise
        
    except Exception as e:
        print(f"\nError uploading dataset: {type(e).__name__}: {str(e)}")
        raise 