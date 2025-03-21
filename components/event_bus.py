class EventBus:
    def __init__(self):
        """Initialize event bus for messaging between components"""
        self.subscribers = {}
        self.event_history = []

    def subscribe(self, event_type, callback):
        """
        Subscribe a callback to an event type
        
        Args:
            event_type (str): Type of event to subscribe to
            callback (function): Function to call when event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def publish(self, event_type, data):
        """
        Publish an event
        
        Args:
            event_type (str): Type of event
            data (dict): Data associated with event
        """
        # Store event in history
        event = {
            "type": event_type,
            "data": data
        }
        self.event_history.append(event)
        
        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(data)

    def get_history(self, event_type=None):
        """
        Get event history
        
        Args:
            event_type (str, optional): Filter by event type
            
        Returns:
            list: Event history
        """
        if event_type:
            return [event for event in self.event_history if event["type"] == event_type]
        return self.event_history