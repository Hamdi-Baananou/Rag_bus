import os
import json
import yaml
from typing import Dict, List, Optional, Union

class PromptManager:
    def __init__(self, prompts_directory: str = "prompts"):
        """
        Initialize the PromptManager to load and manage prompts for batch processing.
        
        Args:
            prompts_directory (str): Directory where prompt templates are stored
        """
        self.prompts_directory = prompts_directory
        self.prompt_templates = {}
        self.default_variables = {}
        
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
            if os.path.isfile(filepath) and (filename.endswith('.json') or filename.endswith('.yaml') or filename.endswith('.yml')):
                try:
                    prompt_name = os.path.splitext(filename)[0]
                    self.prompt_templates[prompt_name] = self._load_prompt_file(filepath)
                except Exception as e:
                    print(f"Error loading prompt '{filename}': {str(e)}")
    
    def _load_prompt_file(self, filepath: str) -> Dict:
        """
        Load a single prompt file
        
        Args:
            filepath (str): Path to the prompt file
            
        Returns:
            Dict: Prompt template definition
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            if filepath.endswith('.json'):
                return json.load(file)
            elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                return yaml.safe_load(file)
    
    def get_available_prompts(self) -> List[str]:
        """
        Get a list of available prompt template names
        
        Returns:
            List[str]: List of available prompt names
        """
        return list(self.prompt_templates.keys())
    
    def get_prompt_info(self, prompt_name: str) -> Dict:
        """
        Get metadata about a specific prompt
        
        Args:
            prompt_name (str): Name of the prompt template
            
        Returns:
            Dict: Prompt metadata
        """
        if prompt_name not in self.prompt_templates:
            raise ValueError(f"Prompt '{prompt_name}' not found")
            
        template = self.prompt_templates[prompt_name]
        return {
            "name": prompt_name,
            "description": template.get("description", "No description provided"),
            "variables": template.get("variables", []),
            "required_variables": template.get("required_variables", [])
        }
    
    def format_prompt(self, prompt_name: str, variables: Dict = None) -> str:
        """
        Format a prompt template with the provided variables
        
        Args:
            prompt_name (str): Name of the prompt template
            variables (Dict, optional): Variables to populate the template
            
        Returns:
            str: Formatted prompt ready for the LLM
        """
        if prompt_name not in self.prompt_templates:
            raise ValueError(f"Prompt '{prompt_name}' not found")
            
        template = self.prompt_templates[prompt_name]
        template_text = template.get("template", "")
        
        # Combine default variables with provided variables
        all_variables = {**self.default_variables}
        if variables:
            all_variables.update(variables)
            
        # Check for required variables
        required_vars = template.get("required_variables", [])
        missing_vars = [var for var in required_vars if var not in all_variables]
        if missing_vars:
            raise ValueError(f"Missing required variables for prompt '{prompt_name}': {', '.join(missing_vars)}")
            
        # Format the template with variables
        for var_name, var_value in all_variables.items():
            placeholder = f"{{{var_name}}}"
            template_text = template_text.replace(placeholder, str(var_value))
            
        return template_text
    
    def set_default_variables(self, variables: Dict) -> None:
        """
        Set default variables to use across all prompts
        
        Args:
            variables (Dict): Default variables
        """
        self.default_variables = variables
    
    def create_prompt(self, prompt_name: str, template: str, description: str = "", 
                      variables: List[str] = None, required_variables: List[str] = None) -> None:
        """
        Create a new prompt template and save it to file
        
        Args:
            prompt_name (str): Name for the new prompt
            template (str): Template text with {variable} placeholders
            description (str, optional): Description of what the prompt does
            variables (List[str], optional): List of variable names used in the template
            required_variables (List[str], optional): List of required variable names
        """
        if not variables:
            variables = []
        if not required_variables:
            required_variables = []
            
        prompt_data = {
            "description": description,
            "template": template,
            "variables": variables,
            "required_variables": required_variables
        }
        
        # Save to JSON file
        filepath = os.path.join(self.prompts_directory, f"{prompt_name}.json")
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(prompt_data, file, indent=2)
            
        # Reload prompts
        self.prompt_templates[prompt_name] = prompt_data
    
    def batch_process(self, prompt_name: str, variable_sets: List[Dict], 
                      callback: callable = None) -> List[Dict]:
        """
        Process a batch of prompts with different variable sets
        
        Args:
            prompt_name (str): Name of the prompt template to use
            variable_sets (List[Dict]): List of variable dictionaries, one for each prompt
            callback (callable, optional): Callback function to call for each formatted prompt
            
        Returns:
            List[Dict]: List of formatted prompts with metadata
        """
        results = []
        
        for i, variables in enumerate(variable_sets):
            try:
                formatted_prompt = self.format_prompt(prompt_name, variables)
                result = {
                    "index": i,
                    "variables": variables,
                    "prompt": formatted_prompt,
                    "status": "success"
                }
                
                # Call the callback function if provided
                if callback and callable(callback):
                    callback_result = callback(formatted_prompt, variables, i)
                    result["callback_result"] = callback_result
                    
            except Exception as e:
                result = {
                    "index": i,
                    "variables": variables,
                    "error": str(e),
                    "status": "error"
                }
                
            results.append(result)
            
        return results


# Example usage in the main file
if __name__ == "__main__":
    # Sample usage example
    prompt_manager = PromptManager()
    
    # Create a sample prompt
    prompt_manager.create_prompt(
        prompt_name="technical_analysis",
        description="Detailed technical analysis of a document with specific focus",
        template="""Analyze the following document excerpt with a focus on {focus_area}.
        
Document: {document_content}

Please provide a detailed analysis covering:
1. Key points related to {focus_area}
2. Technical implications
3. Potential applications
4. Limitations or challenges

If there are specific terms related to {focus_area}, please explain them.
If you're unsure about any aspect, indicate this clearly.
""",
        variables=["document_content", "focus_area"],
        required_variables=["document_content"]
    )
    
    # Use the prompt
    formatted_prompt = prompt_manager.format_prompt(
        "technical_analysis", 
        {
            "document_content": "Sample document content here...",
            "focus_area": "machine learning applications"
        }
    )
    
    print("Available prompts:", prompt_manager.get_available_prompts())
    print("\nSample formatted prompt:\n", formatted_prompt)