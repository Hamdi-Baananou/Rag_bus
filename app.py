# Add this at the very top of the file to fix SQLite version issue
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now continue with your regular imports
import streamlit as st
import os
import re
from components.file_processor import process_pdfs
from components.vector_store import VectorStore
from components.llm_service import LLMService
from components.event_bus import EventBus
from components.evaluation import EvaluationService
from components.prompt_manager import SimplePromptManager

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
if "prompt_manager" not in st.session_state:
    st.session_state.prompt_manager = SimplePromptManager()
if "prompt_batch_queue" not in st.session_state:
    st.session_state.prompt_batch_queue = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload PDFs", "Ask Questions", "Batch Processing", "Evaluation Results", "Prompt Templates"])

# In the tab1 section where you process documents:
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
                    
                    # Get and display vector store stats
                    stats = vector_store.get_stats()
                    
                    # Create metrics display
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Chunks Created", stats["chunks_count"])
                    with col2:
                        st.metric("Total Embeddings Generated", stats["embedding_count"])
                    
                    # Log to event bus
                    st.session_state.event_bus.publish("documents_processed", {
                        "num_documents": len(documents),
                        "document_names": [doc.metadata.get("source") for doc in documents],
                        "chunks_count": stats["chunks_count"],
                        "embedding_count": stats["embedding_count"]
                    })
                    
                    st.session_state.processing_status = f"Successfully processed {len(documents)} documents creating {stats['chunks_count']} chunks with {stats['embedding_count']} embeddings"
                    st.success(st.session_state.processing_status)
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        else:
            st.error("Please upload at least one PDF file")

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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", evaluation_data.get("processed_queries", 0))
        with col2:
            st.metric("Avg. Latency", f"{evaluation_data.get('avg_latency', 0):.2f}s")
        with col3:
            st.metric("Total Tokens Used", evaluation_data.get("total_tokens", 0))
        with col4:
            if st.session_state.vector_store:
                stats = st.session_state.vector_store.get_stats()
                st.metric("Total Chunks", stats["chunks_count"])
        
        # Add Vector Store stats section
        if st.session_state.vector_store:
            st.subheader("Vector Store Statistics")
            stats = st.session_state.vector_store.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", stats["chunks_count"])
            with col2:
                st.metric("Total Embeddings", stats["embedding_count"])
            
            st.info(f"Collection Name: {stats['collection_name']}")
        
        
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

with tab5:
    st.header("Prompt Template Management")
    
    # Sidebar for prompt management
    prompt_action = st.radio("Action", ["Use existing prompts", "Create new prompt"])
    
    if prompt_action == "Use existing prompts":
        available_prompts = st.session_state.prompt_manager.get_available_prompts()
        if not available_prompts:
            st.info("No prompt templates available. Create a new prompt template to get started.")
        else:
            # Add "Select All" button and initialize selected_prompts if not in session state
            if "selected_prompts" not in st.session_state:
                st.session_state.selected_prompts = []
                
            col1, col2 = st.columns([3, 1])
            with col1:
                # Multi-select box for prompts
                selected_prompts = st.multiselect(
                    "Select Prompt Templates", 
                    available_prompts,
                    default=st.session_state.selected_prompts
                )
                # Update session state
                st.session_state.selected_prompts = selected_prompts
                
            with col2:
                # Select All button
                if st.button("Select All"):
                    st.session_state.selected_prompts = available_prompts
                    # Force the page to rerun so the multiselect updates
                    st.experimental_rerun()
            
            # Only proceed if at least one prompt is selected
            if selected_prompts:
                st.subheader("Configure Selected Prompts")
                
                # Allow user to configure variables for each selected prompt
                prompt_configs = []
                
                for selected_prompt in selected_prompts:
                    with st.expander(f"Configure: {selected_prompt}"):
                        prompt_text = st.session_state.prompt_manager.get_prompt(selected_prompt)
                        
                        # Extract all variables except document_content
                        all_variables = list(set(re.findall(r'\{(\w+)\}', prompt_text)))
                        user_variables = [var for var in all_variables if var != 'document_content']
                        
                        if 'document_content' in all_variables:
                            st.info("This prompt contains a {document_content} variable that will be automatically populated from retrieved documents.")
                        
                        # Only show inputs for non-document variables
                        st.subheader("Prompt Variables")
                        variables_dict = {}
                        for var in user_variables:
                            variables_dict[var] = st.text_input(f"{selected_prompt}_{var}", key=f"{selected_prompt}_{var}")
                        
                        # Add this configured prompt to the list
                        prompt_configs.append({
                            "prompt_name": selected_prompt,
                            "variables": variables_dict,
                            "has_document_var": 'document_content' in all_variables,
                            "configured": all(variables_dict.values()) or not user_variables
                        })
                
                # Add button to add these prompts to the queue
                if st.button("Add Selected Prompts to Queue"):
                    # Check if vector store exists
                    if st.session_state.vector_store is None:
                        st.error("Please upload and process documents first before adding prompts to the queue")
                    else:
                        # Check if all prompts are configured
                        unconfigured_prompts = [p["prompt_name"] for p in prompt_configs if not p["configured"]]
                        
                        if unconfigured_prompts:
                            st.error(f"Please complete the configuration for: {', '.join(unconfigured_prompts)}")
                        else:
                            # Add configured prompts to queue
                            if "prompt_batch_queue" not in st.session_state:
                                st.session_state.prompt_batch_queue = []
                                
                            for config in prompt_configs:
                                st.session_state.prompt_batch_queue.append({
                                    "prompt_name": config["prompt_name"],
                                    "variables": config["variables"],
                                    "has_document_var": config["has_document_var"]
                                })
                            
                            st.success(f"Added {len(prompt_configs)} prompts to the queue")
    
    else:  # Create new prompt
        st.subheader("Create New Prompt Template")
        
        new_prompt_name = st.text_input("Prompt Name (no spaces)")
        new_prompt_template = st.text_area("Template", height=200, 
                                         help="Use {variable_name} syntax for variables. Use {document_content} to include retrieved documents.")
        if st.button("Save Prompt Template"):
            if not new_prompt_name or not new_prompt_template:
                st.error("Prompt name and template are required")
            else:
                try:
                    os.makedirs("prompts", exist_ok=True)
                    with open(f"prompts/{new_prompt_name}.yaml", "w", encoding="utf-8") as f:
                        yaml.dump({"prompt": new_prompt_template}, f)
                    st.session_state.prompt_manager._load_prompts()
                    st.success(f"Prompt template '{new_prompt_name}' created successfully")
                except Exception as e:
                    st.error(f"Error creating prompt: {str(e)}")
    
    # Batch Queue Management (if using existing prompts)
    if "prompt_batch_queue" in st.session_state and st.session_state.prompt_batch_queue:
        st.subheader("Batch Prompt Queue")
        st.write(f"Queue size: {len(st.session_state.prompt_batch_queue)}")
        
        for i, item in enumerate(st.session_state.prompt_batch_queue):
            with st.expander(f"Item {i+1}: {item['prompt_name']}"):
                st.write(f"**Prompt:** {item['prompt_name']}")
                st.write("**Variables:**")
                for k, v in item['variables'].items():
                    if len(str(v)) > 50:
                        st.write(f"- {k}: {str(v)[:50]}...")
                    else:
                        st.write(f"- {k}: {v}")
                if item.get('has_document_var', False):
                    st.write("_This prompt will use retrieved document content_")
        
        if st.button("Process Batch Queue") and st.session_state.vector_store:
            if not api_key:
                st.error("Please enter your OpenRouter API key in the sidebar")
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
                
                # Process queue
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                
                for i, item in enumerate(st.session_state.prompt_batch_queue):
                    status_text.text(f"Processing prompt {i+1}/{len(st.session_state.prompt_batch_queue)}...")
                    
                    try:
                        # Get the prompt template
                        prompt_template = st.session_state.prompt_manager.get_prompt(item['prompt_name'])
                        
                        # If it has document_content variable, we need to retrieve docs and format
                        if item.get('has_document_var', False):
                            # Construct a question from the variables to use for retrieval
                            # You might need to adjust this logic based on your specific needs
                            question = " ".join(item['variables'].values()) if item['variables'] else "retrieve relevant documents"
                            
                            # Get relevant documents
                            docs = retriever.get_relevant_documents(question)
                            doc_content = "\n\n".join([doc.page_content for doc in docs])
                            
                            # Add document content to variables
                            variables = item['variables'].copy()
                            variables['document_content'] = doc_content
                            
                            # Format the prompt with all variables including document content
                            formatted_prompt = st.session_state.prompt_manager.format_prompt(
                                item['prompt_name'], variables
                            )
                        else:
                            # Just format with the user-provided variables
                            formatted_prompt = st.session_state.prompt_manager.format_prompt(
                                item['prompt_name'], item['variables']
                            )
                        
                        # Get answer from LLM
                        result = llm_service.ask_question(formatted_prompt, retriever)
                        
                        batch_results.append({
                            "prompt_name": item['prompt_name'],
                            "variables": item['variables'],
                            "result": result,
                            "timestamp": st.session_state.evaluation_service.get_timestamp()
                        })
                        
                        # Evaluate answer if enabled
                        if enable_answer_eval:
                            st.session_state.evaluation_service.evaluate_answer(
                                formatted_prompt, result
                            )
                            
                    except Exception as e:
                        batch_results.append({
                            "prompt_name": item['prompt_name'],
                            "variables": item['variables'],
                            "result": f"Error: {str(e)}",
                            "timestamp": st.session_state.evaluation_service.get_timestamp()
                        })
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(st.session_state.prompt_batch_queue))
                
                # Store results and clear queue
                st.session_state.batch_results = batch_results
                st.session_state.prompt_batch_queue = []
                status_text.text("All prompts processed!")
                
                # Display results
                st.subheader("Batch Results")
                for i, result in enumerate(batch_results):
                    with st.expander(f"Result {i+1}: {result['prompt_name']}"):
                        st.write(f"**Prompt Template:** {result['prompt_name']}")
                        st.write("**Variables:** ", ", ".join(result['variables'].keys()))
                        st.markdown(f"**Result:**\n\n{result['result']}")
                        st.text(f"Processed at: {result['timestamp']}")

    # Display previous batch results
    if "batch_results" in st.session_state and st.session_state.batch_results:
        st.subheader("Previous Batch Results")
        for i, result in enumerate(st.session_state.batch_results):
            with st.expander(f"Result {i+1}: {result['prompt_name']}"):
                st.write(f"**Prompt Template:** {result['prompt_name']}")
                st.write("**Variables:** ", ", ".join(result['variables'].keys()))
                st.markdown(f"**Result:**\n\n{result['result']}")
                st.text(f"Processed at: {result['timestamp']}")

# Footer
st.markdown("---")
st.markdown("Advanced PDF QA System with Evaluation Capabilities")