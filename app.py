import streamlit as st
import os
from components.file_processor import process_pdfs
from components.vector_store import VectorStore
from components.llm_service import LLMService
from components.event_bus import EventBus
from components.evaluation import EvaluationService

# Configure page
st.set_page_config(
    page_title="PDF QA System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state for storing application state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "event_bus" not in st.session_state:
    st.session_state.event_bus = EventBus()
if "evaluation_service" not in st.session_state:
    st.session_state.evaluation_service = EvaluationService(st.session_state.event_bus)
if "extraction_queue" not in st.session_state:
    st.session_state.extraction_queue = []
if "extraction_results" not in st.session_state:
    st.session_state.extraction_results = []
if "processing_status" not in st.session_state:
    st.session_state.processing_status = ""

# App Header
st.title("Advanced PDF Question Answering System")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenRouter API Key", type="password", key="api_key")
    
    # LLM Model Selection
    model_options = {
        "qwen/qwq-32b:free": "Qwen 32B (Free)",
        "anthropic/claude-3-opus-20240229": "Claude 3 Opus",
        "anthropic/claude-3-sonnet-20240229": "Claude 3 Sonnet"
    }
    selected_model = st.selectbox("Select LLM Model", 
                                  options=list(model_options.keys()),
                                  format_func=lambda x: model_options[x],
                                  key="selected_model")
    
    # Extraction parameters
    st.subheader("Extraction Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=256, max_value=4096, value=2048, step=256)
    
    # Evaluation options
    st.subheader("Evaluation Options")
    enable_answer_eval = st.checkbox("Evaluate Answer Quality", value=True)
    enable_latency_eval = st.checkbox("Measure Response Latency", value=True)
    enable_token_tracking = st.checkbox("Track Token Usage", value=True)

# Main functionality in tabs
tab1, tab2, tab3, tab4 = st.tabs(["Upload PDFs", "Ask Questions", "Batch Processing", "Evaluation Results"])

with tab1:
    st.header("Upload and Process Documents")
    uploaded_files = st.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents", key="process_docs"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files temporarily
                    temp_paths = []
                    for file in uploaded_files:
                        temp_path = f"temp/{file.name}"
                        os.makedirs("temp", exist_ok=True)
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        temp_paths.append(temp_path)
                    
                    # Process documents
                    documents = process_pdfs(temp_paths)
                    
                    # Initialize vector store
                    vector_store = VectorStore()
                    st.session_state.vector_store = vector_store
                    retriever = vector_store.setup_chroma(documents)
                    
                    # Log to event bus
                    st.session_state.event_bus.publish("documents_processed", {
                        "num_documents": len(documents),
                        "document_names": [doc.metadata.get("source") for doc in documents]
                    })
                    
                    st.session_state.processing_status = f"Successfully processed {len(documents)} documents"
                    st.success(st.session_state.processing_status)
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        else:
            st.error("Please upload at least one PDF file")
    
    # Display processing status
    if st.session_state.processing_status:
        st.info(st.session_state.processing_status)

with tab2:
    st.header("Ask Questions")
    
    if st.session_state.vector_store is None:
        st.warning("Please upload and process documents first")
    else:
        question = st.text_input("Your Question", placeholder="What would you like to know from the documents?")
        
        if st.button("Get Answer", key="ask_question"):
            if not question:
                st.error("Please enter a question")
            elif not api_key:
                st.error("Please enter your OpenRouter API key in the sidebar")
            else:
                with st.spinner("Getting answer..."):
                    try:
                        # Create LLM service
                        llm_service = LLMService(
                            api_key=api_key,
                            model_name=selected_model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            event_bus=st.session_state.event_bus
                        )
                        
                        # Get retriever from vector store
                        retriever = st.session_state.vector_store.get_retriever()
                        
                        # Get answer
                        result = llm_service.ask_question(question, retriever)
                        
                        # Display answer
                        st.subheader("Answer:")
                        st.markdown(result)
                        
                        # Evaluate answer if enabled
                        if enable_answer_eval:
                            st.session_state.evaluation_service.evaluate_answer(question, result)
                            
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")

with tab3:
    st.header("Batch Question Processing")
    
    if st.session_state.vector_store is None:
        st.warning("Please upload and process documents first")
    else:
        # Input for batch questions
        batch_input = st.text_area("Enter Multiple Questions (one per line)", 
                                 height=150, 
                                 placeholder="Question 1\nQuestion 2\nQuestion 3")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add to Queue", key="add_to_queue"):
                if not batch_input:
                    st.error("Please enter at least one question")
                else:
                    questions = [q.strip() for q in batch_input.split("\n") if q.strip()]
                    st.session_state.extraction_queue.extend(questions)
                    st.success(f"Added {len(questions)} questions to the queue")
        
        with col2:
            if st.button("Process Queue", key="process_queue"):
                if not api_key:
                    st.error("Please enter your OpenRouter API key in the sidebar")
                elif not st.session_state.extraction_queue:
                    st.error("Queue is empty. Add questions first.")
                else:
                    # Create LLM service
                    llm_service = LLMService(
                        api_key=api_key,
                        model_name=selected_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        event_bus=st.session_state.event_bus
                    )
                    
                    # Get retriever from vector store
                    retriever = st.session_state.vector_store.get_retriever()
                    
                    # Process each question in the queue
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, question in enumerate(st.session_state.extraction_queue):
                        status_text.text(f"Processing question {i+1}/{len(st.session_state.extraction_queue)}: {question[:50]}...")
                        
                        try:
                            # Get answer
                            result = llm_service.ask_question(question, retriever)
                            
                            # Add result to results list
                            st.session_state.extraction_results.append({
                                "question": question,
                                "answer": result,
                                "timestamp": st.session_state.evaluation_service.get_timestamp()
                            })
                            
                            # Evaluate answer if enabled
                            if enable_answer_eval:
                                st.session_state.evaluation_service.evaluate_answer(question, result)
                                
                        except Exception as e:
                            st.session_state.extraction_results.append({
                                "question": question,
                                "answer": f"Error: {str(e)}",
                                "timestamp": st.session_state.evaluation_service.get_timestamp()
                            })
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(st.session_state.extraction_queue))
                    
                    # Clear queue after processing
                    st.session_state.extraction_queue = []
                    status_text.text("All questions processed!")
                    
        # Display current queue
        if st.session_state.extraction_queue:
            st.subheader("Current Queue")
            for i, q in enumerate(st.session_state.extraction_queue):
                st.text(f"{i+1}. {q}")
        
        # Display extraction results
        if st.session_state.extraction_results:
            st.subheader("Processing Results")
            for i, result in enumerate(st.session_state.extraction_results):
                with st.expander(f"Result {i+1}: {result['question'][:50]}..."):
                    st.markdown(f"**Question:** {result['question']}")
                    st.markdown(f"**Answer:** {result['answer']}")
                    st.text(f"Processed at: {result['timestamp']}")

with tab4:
    st.header("Evaluation Results")
    
    # Display evaluation metrics if available
    evaluation_data = st.session_state.evaluation_service.get_evaluation_data()
    
    if evaluation_data and evaluation_data.get("processed_queries", 0) > 0:
        st.subheader("System Performance")
        
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", evaluation_data.get("processed_queries", 0))
        with col2:
            st.metric("Avg. Latency", f"{evaluation_data.get('avg_latency', 0):.2f}s")
        with col3:
            st.metric("Total Tokens Used", evaluation_data.get("total_tokens", 0))
        
        # Add more detailed evaluation data as needed
        if evaluation_data.get("answer_quality_scores"):
            st.subheader("Answer Quality")
            st.line_chart(evaluation_data.get("answer_quality_scores"))
        
        if evaluation_data.get("latency_data"):
            st.subheader("Response Latency")
            st.line_chart(evaluation_data.get("latency_data"))
            
        # Add option to export evaluation data
        if st.button("Export Evaluation Data"):
            # Code to export evaluation data as CSV or JSON
            st.download_button(
                label="Download Evaluation Data",
                data=st.session_state.evaluation_service.export_evaluation_data(),
                file_name="evaluation_data.json",
                mime="application/json"
            )
    else:
        st.info("No evaluation data available yet. Process some questions to generate evaluation metrics.")

# Footer
st.markdown("---")
st.markdown("Advanced PDF QA System with Evaluation Capabilities")