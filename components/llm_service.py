import requests
import re
import time

class LLMService:
    def __init__(self, api_key, model_name="qwen/qwq-32b:free", temperature=0.3, max_tokens=2048, event_bus=None):
        """
        Initialize the LLM service
        
        Args:
            api_key (str): OpenRouter API key
            model_name (str): Model to use
            temperature (float): Temperature for generation
            max_tokens (int): Maximum tokens to generate
            event_bus (EventBus): Event bus for tracking metrics
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.event_bus = event_bus

    def ask_question(self, question, retriever):
        """
        Ask a question with context from the retriever
        
        Args:
            question (str): Question to ask
            retriever: Document retriever
            
        Returns:
            str: Answer from the LLM
        """
        start_time = time.time()
        
        # Retrieve relevant documents
        results = retriever.invoke(question)
        if not results:
            return "No relevant information found in documents"

        # Constructing the context from retrieved documents
        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
            for doc in results
        ])

        # Formulating the API prompt
        prompt = f"""Answer based on context:
        {context}

        Question: {question}

        If unsure, say "I don't have enough information".
        Include source references when possible."""

        # Defining API request payload
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "top_p": 1,
            "top_k": 40,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit-app.com",  # Update for your app
            "X-Title": "PDF QA Assistant"
        }

        # Sending request to OpenRouter API
        response = requests.post(self.api_url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"API Error ({response.status_code}): {response.text[:200]}...")

        response_data = response.json()

        # Handle OpenRouter's response format
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            raise Exception("Invalid response format from API")

        first_choice = response_data['choices'][0]
        message_content = first_choice.get('message', {}).get('content', '')

        if not message_content:
            raise Exception("Empty response from API")

        # Clean up the response
        cleaned_answer = re.sub(r'\\boxed{', '', message_content)
        cleaned_answer = re.sub(r'\\[^\s]+', '', cleaned_answer)
        cleaned_answer = cleaned_answer.replace('\\n', '\n')

        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Publish metrics to event bus if available
        if self.event_bus:
            self.event_bus.publish("llm_response", {
                "question": question,
                "answer": cleaned_answer.strip(),
                "latency": elapsed_time,
                "model": self.model_name,
                "token_usage": response_data.get("usage", {})
            })

        return cleaned_answer.strip()