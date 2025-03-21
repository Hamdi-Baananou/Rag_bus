import time
import json
from datetime import datetime

class EvaluationService:
    def __init__(self, event_bus):
        """
        Initialize evaluation service
        
        Args:
            event_bus (EventBus): Event bus for receiving events
        """
        self.event_bus = event_bus
        self.metrics = {
            "processed_queries": 0,
            "total_latency": 0,
            "avg_latency": 0,
            "total_tokens": 0,
            "latency_data": {},
            "token_usage_data": {},
            "answer_quality_scores": {},
            "sources_used": set(),
            "errors": []
        }
        
        # Subscribe to events
        self.event_bus.subscribe("llm_response", self.handle_llm_response)
        self.event_bus.subscribe("documents_processed", self.handle_documents_processed)
    
    def handle_llm_response(self, data):
        """
        Handle LLM response event
        
        Args:
            data (dict): Event data
        """
        self.metrics["processed_queries"] += 1
        
        # Track latency
        if "latency" in data:
            self.metrics["total_latency"] += data["latency"]
            self.metrics["avg_latency"] = self.metrics["total_latency"] / self.metrics["processed_queries"]
            timestamp = self.get_timestamp()
            self.metrics["latency_data"][timestamp] = data["latency"]
        
        # Track token usage
        if "token_usage" in data:
            token_usage = data["token_usage"]
            if token_usage:
                total_tokens = token_usage.get("total_tokens", 0)
                self.metrics["total_tokens"] += total_tokens
                timestamp = self.get_timestamp()
                self.metrics["token_usage_data"][timestamp] = total_tokens
    
    def handle_documents_processed(self, data):
        """
        Handle documents processed event
        
        Args:
            data (dict): Event data
        """
        if "document_names" in data:
            self.metrics["sources_used"].update(data["document_names"])
    
    def evaluate_answer(self, question, answer):
        """
        Evaluate answer quality (placeholder for more advanced evaluation)
        
        Args:
            question (str): Question
            answer (str): Answer
        """
        # Simple evaluation based on answer length and source citations
        score = 0
        
        # Length-based scoring
        if len(answer) > 200:
            score += 5
        elif len(answer) > 100:
            score += 3
        elif len(answer) > 50:
            score += 1
        
        # Source citation scoring
        if "Source:" in answer:
            score += 3
        
        # Track score
        timestamp = self.get_timestamp()
        self.metrics["answer_quality_scores"][timestamp] = score
        
        return score
    
    def get_evaluation_data(self):
        """
        Get evaluation data
        
        Returns:
            dict: Evaluation metrics
        """
        return self.metrics
    
    def export_evaluation_data(self):
        """
        Export evaluation data as JSON
        
        Returns:
            str: JSON string of evaluation data
        """
        # Convert sets to lists for JSON serialization
        export_data = self.metrics.copy()
        export_data["sources_used"] = list(export_data["sources_used"])
        
        return json.dumps(export_data, indent=2)
    
    def get_timestamp(self):
        """
        Get current timestamp
        
        Returns:
            str: Current timestamp
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")