import os
import yaml
from typing import Dict, List

class SimplePromptManager:
    def __init__(self, prompts_directory: str = "prompts"):
        """
        Initialize the SimplePromptManager to load prompts from YAML files.
       
        Args:
            prompts_directory (str): Directory where prompt templates are stored
        """
        self.prompts_directory = prompts_directory
        self.prompts = {}
       
        # Create prompts directory if it doesn't exist
        os.makedirs(self.prompts_directory, exist_ok=True)
       
        # Load all available prompts
        self._load_prompts()
   
    def _load_prompts(self) -> None:
        """Load all prompt templates from the prompts directory"""
        if not os.path.exists(self.prompts_directory):
            return
           
        for filename in os.listdir(self.prompts_directory):
            filepath = os.path.join(self.prompts_directory, filename)
            if os.path.isfile(filepath) and filename.endswith('.yaml'):
                try:
                    prompt_name = os.path.splitext(filename)[0]
                    self.prompts[prompt_name] = self._load_prompt_file(filepath)
                except Exception as e:
                    print(f"Error loading prompt '{filename}': {str(e)}")
   
    def _load_prompt_file(self, filepath: str) -> str:
        """
        Load a single prompt file
       
        Args:
            filepath (str): Path to the prompt file
           
        Returns:
            str: Prompt template
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            prompt_data = yaml.safe_load(file)
            return prompt_data.get('prompt', '')
   
    def get_available_prompts(self) -> List[str]:
        """
        Get a list of available prompt names
       
        Returns:
            List[str]: List of available prompt names
        """
        return list(self.prompts.keys())
   
    def get_prompt(self, prompt_name: str) -> str:
        """
        Get a specific prompt by name
       
        Args:
            prompt_name (str): Name of the prompt
           
        Returns:
            str: The prompt text
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")
           
        return self.prompts[prompt_name]
   
    def format_prompt(self, prompt_name: str, variables: Dict = None) -> str:
        """
        Format a prompt with the provided variables
       
        Args:
            prompt_name (str): Name of the prompt
            variables (Dict, optional): Variables to populate in the prompt
           
        Returns:
            str: Formatted prompt
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")
           
        template = self.prompts[prompt_name]
       
        # If no variables provided, return the template as is
        if not variables:
            return template
           
        # Format the template with variables
        formatted_prompt = template
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            formatted_prompt = formatted_prompt.replace(placeholder, str(var_value))
           
        return formatted_prompt

# Example usage
if __name__ == "__main__":
    # Initialize the prompt manager
    prompt_manager = SimplePromptManager()
   
    # Show available prompts
    available_prompts = prompt_manager.get_available_prompts()
    print("Available prompts:", available_prompts)
   
    # Example of how to use a prompt if any are available
    if available_prompts:
        example_prompt = available_prompts[0]
        formatted_prompt = prompt_manager.format_prompt(
            example_prompt,
            {"document_content": "Sample document text"}
        )
        print(f"\nFormatted prompt '{example_prompt}':\n", formatted_prompt)