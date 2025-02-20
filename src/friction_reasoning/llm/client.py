"""LiteLLM client implementation for agent reasoning."""

import subprocess
from typing import Dict, Optional, Any, List
from litellm import acompletion
from pydantic import BaseModel

class LLMResponse(BaseModel):
    """Structured response from LLM."""
    thought_stream: str
    friction_points: list[Dict[str, str]]
    metadata: Dict[str, Any]

class OllamaModelNotFoundError(Exception):
    """Raised when an Ollama model is not found."""
    pass

class LLMClient:
    """Wrapper for LiteLLM to handle model interactions."""
    
    def __init__(self, model: str = "ollama/mistral-small", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        
    async def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Get completion from the model asynchronously."""
        messages: List[Dict[str, str]] = []
        
        if system:
            messages.append({"role": "system", "content": system})
            
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return f"Error: {str(e)}"
            
    async def stream_complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Get streaming completion from the model asynchronously."""
        messages: List[Dict[str, str]] = []
        
        if system:
            messages.append({"role": "system", "content": system})
            
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True
            )
            
            full_response = ""
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()  # New line after streaming
            return full_response
            
        except Exception as e:
            print(f"Error streaming from LLM: {e}")
            return f"Error: {str(e)}"

    def _verify_ollama_model(self, model_name: str) -> None:
        """Verify that an Ollama model is available and up to date.
        
        Args:
            model_name: Name of the Ollama model to verify
            
        Raises:
            OllamaModelNotFoundError: If model is not found
            subprocess.CalledProcessError: If ollama command fails
        """
        try:
            # Run ollama list and parse output
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Parse the output into rows and columns
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:  # Need header and at least one model
                raise OllamaModelNotFoundError(
                    f"Model {model_name} not found. Please pull it using 'ollama pull {model_name}'"
                )
                
            # Parse header and find column indices
            headers = [h.strip() for h in lines[0].split()]
            name_idx = headers.index("NAME")
            modified_idx = headers.index("MODIFIED")
            
            # Look for our model
            model_found = False
            for line in lines[1:]:
                cols = line.split()
                if cols[name_idx].startswith(model_name):
                    model_found = True
                    # Check modification time
                    modified_str = " ".join(cols[modified_idx:modified_idx+2])  # "X hours/days ago"
                    self._check_model_freshness(model_name, modified_str)
                    break
            
            if not model_found:
                raise OllamaModelNotFoundError(
                    f"Model {model_name} not found. Please pull it using 'ollama pull {model_name}'"
                )
                    
        except subprocess.CalledProcessError as e:
            raise OllamaModelNotFoundError(
                f"Failed to verify Ollama model: {str(e)}"
            )
    
    def _check_model_freshness(self, model_name: str, modified_str: str) -> None:
        """Check if the model was modified recently and warn if it's old.
        
        Args:
            model_name: Name of the model
            modified_str: String like "X hours ago" or "X days ago"
        """
        try:
            # Parse the modification time
            value = int(modified_str.split()[0])
            unit = modified_str.split()[1]
            
            if "day" in unit and value > 7:  # Older than a week
                print(f"Warning: Model {model_name} was last modified {modified_str}.")
                print(f"Consider updating with 'ollama pull {model_name}'")
                
        except (ValueError, IndexError):
            # If we can't parse the time, just skip the freshness check
            pass
    
    async def generate_thought_stream(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        temperature: float = 0.7
    ) -> LLMResponse:
        """Generate a thought stream response from the LLM.
        
        Args:
            prompt: The input prompt for reasoning
            context: Optional context from previous responses
            temperature: Sampling temperature (default: 0.7)
            
        Returns:
            Structured response containing thought stream and friction points
        """
        # Construct the full prompt with context
        full_prompt = self._build_prompt(prompt, context)
        
        # Get completion from LiteLLM with Ollama configuration
        response = await acompletion(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=temperature,
            api_base=self.api_base
        )
        
        # Parse and structure the response
        return self._parse_response(response.choices[0].message.content)
    
    def _build_prompt(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Build the full prompt with context if provided."""
        if not context:
            return prompt
            
        # Add context from previous agent responses
        context_str = "\nPrevious thoughts:\n" + context.get("raw_thought_stream", "")
        return prompt + context_str
    
    def _parse_response(self, raw_response: str) -> LLMResponse:
        """Parse raw LLM response into structured format.
        
        Extracts thought stream and friction points from the raw response.
        Friction points are identified by text between *asterisks*.
        """
        # Extract friction points (text between asterisks)
        friction_points = []
        current_pos = 0
        while True:
            start = raw_response.find("*", current_pos)
            if start == -1:
                break
            end = raw_response.find("*", start + 1)
            if end == -1:
                break
            marker = raw_response[start:end + 1]
            friction_points.append({
                "type": "friction_point",
                "marker": marker
            })
            current_pos = end + 1
        
        return LLMResponse(
            thought_stream=raw_response,
            friction_points=friction_points,
            metadata={
                "model": self.model,
                "api_base": self.api_base
            }
        ) 