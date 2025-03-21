# Advanced PDF QA System

A Streamlit application for extracting information from PDF documents using Large Language Models (LLMs) with comprehensive evaluation capabilities.

## Features

- **PDF Processing**: Upload and process multiple PDF documents
- **Vector Storage**: Store and retrieve document embeddings using ChromaDB
- **Question Answering**: Ask questions about the content of processed documents
- **Batch Processing**: Queue multiple questions for sequential processing
- **Evaluation System**: Track and analyze performance metrics
- **Event-Driven Architecture**: Using an event bus for loose coupling between components

## Project Structure

```
.
├── app.py                    # Main Streamlit application
├── requirements.txt          # Project dependencies
├── temp/                     # Temporary storage for uploaded files
└── components/
    ├── __init__.py           # Package initialization
    ├── file_processor.py     # PDF processing functions
    ├── vector_store.py       # Vector database management
    ├── llm_service.py        # LLM interaction service
    ├── event_bus.py          # Event messaging system
    └── evaluation.py         # Evaluation and metrics tracking
```

## Setup & Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

1. **Upload PDFs**: Begin by uploading one or more PDF documents
2. **Process Documents**: Click "Process Documents" to extract and embed content
3. **Ask Questions**: Enter questions about the document content
4. **Queue Questions**: For batch processing, add multiple questions to the queue
5. **View Evaluations**: Monitor system performance metrics in the Evaluation tab

## Configuration

- **API Key**: Enter your OpenRouter API key in the sidebar
- **Model Selection**: Choose from available LLM models
- **Parameters**: Adjust temperature and max tokens for generation
- **Evaluation Options**: Enable/disable various evaluation metrics

## Evaluation Metrics

The system tracks several performance metrics:

- Response latency
- Token usage
- Answer quality scores (basic implementation)
- Document source tracking

## Future Enhancements

- Advanced answer quality evaluation
- Multi-modal document support
- Custom evaluation pipelines
- User feedback collection
- Additional document preprocessing options

## License

MIT

# Advanced PDF QA System - Implementation Plan

## Project Structure
```
.
├── app.py                    # Main Streamlit application
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── temp/                     # Temporary storage for uploaded files
└── components/
    ├── __init__.py           # Package initialization
    ├── file_processor.py     # PDF processing functions
    ├── vector_store.py       # Vector database management
    ├── llm_service.py        # LLM interaction service
    ├── event_bus.py          # Event messaging system
    └── evaluation.py         # Evaluation and metrics tracking
```

## Implementation Steps

1. **Set Up Project Structure**
   - Create directory structure as outlined above
   - Initialize python modules with appropriate imports

2. **Implement Core Components**
   - Components are designed with single responsibility principle
   - Each component has clear inputs and outputs
   - Event bus enables loose coupling between components

3. **Build Streamlit Interface**
   - Create intuitive UI with tabs for different functions
   - Add configuration options in sidebar
   - Implement session state management

4. **Implement Evaluation System**
   - Build event-driven metric collection
   - Create visualization for evaluation data
   - Enable export of evaluation results

5. **Testing**
   - Test with different PDF types and sizes
   - Verify batch processing functionality
   - Check evaluation metrics accuracy

## Component Responsibilities

### File Processor
- Handle PDF uploads and processing
- Extract and clean text content
- Create document objects for vector storage

### Vector Store
- Manage document embeddings
- Create and maintain vector database
- Provide retrieval functionality

### LLM Service
- Handle API communication with OpenRouter
- Format prompts with context
- Process responses and handle errors

### Event Bus
- Enable publish-subscribe messaging
- Decouple components for easier testing
- Track system events for evaluation

### Evaluation Service
- Collect performance metrics
- Analyze answer quality (expandable)
- Generate reports and visualizations

## Future Enhancements

1. **Advanced Evaluation**
   - Implement multiple evaluation strategies
   - Add reference-based evaluation
   - Create custom evaluation pipelines

2. **Extraction Enhancements**
   - Support for structured data extraction
   - Template-based extraction for specific document types
   - Custom extraction rules

3. **User Feedback System**
   - Allow users to rate answer quality
   - Collect feedback for model improvement
   - Track user satisfaction metrics

4. **Document Processing Improvements**
   - Add OCR for image-based PDFs
   - Support additional document formats
   - Implement chunking strategies for large documents