from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
from chromadb import Client
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import uuid
import hashlib
from datetime import datetime
import os
import time
import streamlit as st

class VectorStore:
    def __init__(self):
        """Initialize the vector store with embedding function and Chroma client"""
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.client = Client(Settings())
        self.collection_name = f"pdf_qa_{str(uuid.uuid4())[:8]}"  # Create unique collection for each session
        self.retriever = None
        self.chunks_count = 0
        self.embedding_count = 0
        self.documents = []
        self._query_history = []
        
        # Default chunking settings
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.separator = "\n"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[self.separator, ".", "!", "?", ",", " ", ""]
        )

    def setup_advanced_chunking(self, chunk_size=500, chunk_overlap=50, separator="\n"):
        """
        Configure advanced chunking options with character-based approach
        
        Args:
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
            separator (str): Preferred separator for chunks
            
        Returns:
            dict: Current chunking configuration
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        # Use RecursiveCharacterTextSplitter for better semantic chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[self.separator, ".", "!", "?", ",", " ", ""]
        )
        
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separator": self.separator
        }
    
    def set_embedding_model(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        """
        Change the embedding model dynamically
        
        Args:
            model_name (str): Name of the HuggingFace model to use
            device (str): Device to run the model on (cpu/cuda)
            
        Returns:
            dict: Current model configuration
        """
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        
        # Return current model info
        return {"model": model_name, "device": device}
    
    def configure_retriever(self, k=3, search_type="similarity", score_threshold=None, 
                            filter_criteria=None, include_metadata=True):
        """
        Configure retrieval parameters for better search results
        
        Args:
            k (int): Number of documents to retrieve
            search_type (str): Type of search (similarity/mmr)
            score_threshold (float): Minimum similarity score
            filter_criteria (dict): Metadata filter criteria
            include_metadata (bool): Include metadata in results
            
        Returns:
            Retriever: Configured retriever object
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Process documents first.")
            
        search_kwargs = {
            "k": k,
            "score_threshold": score_threshold,
            "filter": filter_criteria
        }
        
        # Remove None values
        search_kwargs = {k: v for k, v in search_kwargs.items() if v is not None}
        
        self.retriever = Chroma(
            collection_name=self.collection_name,
            client=self.client,
            embedding_function=self.embedding_function
        ).as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return self.retriever
    
    def process_documents(self, documents, use_advanced_chunking=False):
        """
        Process documents with optional advanced chunking
        
        Args:
            documents (list): List of Document objects
            use_advanced_chunking (bool): Whether to use the text splitter
            
        Returns:
            list: Processed documents
        """
        if use_advanced_chunking and hasattr(self, 'text_splitter'):
            # Split documents using the configured text splitter
            all_splits = []
            for doc in documents:
                doc_splits = self.text_splitter.split_documents([doc])
                all_splits.extend(doc_splits)
            
            # Update and return the processed documents
            processed_docs = all_splits
        else:
            # Use the documents as-is
            processed_docs = documents
        
        # Store documents for potential hybrid search
        self.documents = processed_docs
        return processed_docs
    
    def setup_chroma(self, documents, use_advanced_chunking=False):
        """
        Set up Chroma with documents
        
        Args:
            documents (list): List of Document objects
            use_advanced_chunking (bool): Whether to use advanced chunking
            
        Returns:
            Retriever: Chroma retriever object
        """
        # Process documents with optional advanced chunking
        processed_docs = self.process_documents(documents, use_advanced_chunking)
        
        # Delete existing collection if present
        existing_collections = self.client.list_collections()
        collection_names = [col.name for col in existing_collections]
        if self.collection_name in collection_names:
            self.client.delete_collection(self.collection_name)

        collection = self.client.create_collection(self.collection_name)

        # Update chunks count
        self.chunks_count = len(processed_docs)
        st.info(f"Created {self.chunks_count} chunks from the documents")

        # Batch process embeddings
        texts = [self.enrich_metadata(doc, doc.metadata).page_content for doc in processed_docs]
        
        # Track embedding time
        start_time = time.time()
        embeddings = self.embedding_function.embed_documents(texts)
        embedding_time = time.time() - start_time
        
        # Update embeddings count
        self.embedding_count = len(embeddings)
        st.info(f"Generated {self.embedding_count} embeddings in {embedding_time:.2f} seconds")

        # Add documents with metadata
        for idx, (text, doc) in enumerate(zip(texts, processed_docs)):
            collection.add(
                documents=[text],
                metadatas=[doc.metadata],
                embeddings=[embeddings[idx]],
                ids=[str(idx)]
            )

        self.retriever = Chroma(
            collection_name=self.collection_name,
            client=self.client,
            embedding_function=self.embedding_function
        ).as_retriever(search_kwargs={"k": 3})
        
        return self.retriever
    
    def search_with_metadata_filter(self, query, filter_criteria=None, k=3):
        """
        Search with specific metadata filters
        
        Args:
            query (str): Search query
            filter_criteria (dict): Metadata filter criteria
            k (int): Number of documents to retrieve
            
        Returns:
            list: Retrieved documents
        
        Example: 
            filter_criteria={"source": {"$eq": "specific_doc.pdf"}}
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Process documents first.")
            
        start_time = time.time()
        docs = Chroma(
            collection_name=self.collection_name,
            client=self.client,
            embedding_function=self.embedding_function
        ).as_retriever(
            search_kwargs={
                "k": k,
                "filter": filter_criteria
            }
        ).get_relevant_documents(query)
        
        retrieval_time = time.time() - start_time
        
        # Log the query for performance tracking
        self.log_query(query, docs, retrieval_time)
        
        return docs
    
    def setup_hybrid_search(self, k_vector=3, k_bm25=3, vector_weight=0.7, bm25_weight=0.3):
        """
        Set up hybrid search combining BM25 and vector search
        
        Args:
            k_vector (int): Number of documents to retrieve from vector search
            k_bm25 (int): Number of documents to retrieve from BM25
            vector_weight (float): Weight for vector search results
            bm25_weight (float): Weight for BM25 results
            
        Returns:
            EnsembleRetriever: Hybrid retriever object
        """
        if not self.documents:
            raise ValueError("No documents available. Process documents first.")
            
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(
            self.documents,
            k=k_bm25
        )
        
        # Configure vector retriever
        vector_retriever = Chroma(
            collection_name=self.collection_name,
            client=self.client,
            embedding_function=self.embedding_function
        ).as_retriever(search_kwargs={"k": k_vector})
        
        # Create ensemble retriever
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[vector_weight, bm25_weight]
        )
        
        return self.hybrid_retriever
    
    def add_reranker(self, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3, cohere_api_key=None):
        """
    Add a reranker to improve retrieval precision
    
    Args:
        reranker_model (str): Model name for reranking
        top_n (int): Number of documents to keep after reranking
        cohere_api_key (str): Cohere API key for authentication
        
    Returns:
        ContextualCompressionRetriever: Reranking retriever
    """
        if self.retriever is None:
            raise ValueError("Base retriever not initialized. Process documents first.")
        try:
            api_key = cohere_api_key or "n7aGb2q854nHoMl3lqzmEaKlqgny9QhwKgkQ1JPB"
            compressor = CohereRerank(top_n=top_n, cohere_api_key=api_key)
            
            self.reranking_retriever = ContextualCompressionRetriever(
                base_retriever=self.retriever,
                base_compressor=compressor
        )
        
            return self.reranking_retriever
        
        except Exception as e:
            st.warning(f"Error setting up reranker: {str(e)}. Make sure you have the required dependencies installed.")
        # Return the original retriever if reranker setup fails
            return self.retriever
    
    def enrich_metadata(self, document, metadata):
        """
        Enrich document metadata with additional information
        
        Args:
            document: Document object
            metadata (dict): Existing metadata
            
        Returns:
            document: Document with enriched metadata
        """
        # Create a copy to avoid modifying the original
        enriched_metadata = metadata.copy()
        
        # Add document hash for deduplication
        content_hash = hashlib.md5(document.page_content.encode()).hexdigest()
        enriched_metadata["content_hash"] = content_hash
        
        # Add additional metadata
        enriched_metadata["chunk_id"] = str(uuid.uuid4())
        enriched_metadata["timestamp"] = datetime.now().isoformat()
        enriched_metadata["char_count"] = len(document.page_content)
        enriched_metadata["word_count"] = len(document.page_content.split())
        
        # Update the document's metadata
        document.metadata = enriched_metadata
        
        return document
    
    def persist_vector_store(self, persist_directory="./chroma_db"):
        """
        Persist the vector store to disk
        
        Args:
            persist_directory (str): Directory to persist the vector store
            
        Returns:
            dict: Information about the persisted store
        """
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create a persistent Chroma instance
        persisted_client = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=persist_directory
        )
        
        # Persist to disk
        persisted_client.persist()
        
        return {
            "persist_directory": persist_directory, 
            "collection": self.collection_name,
            "chunks_count": self.chunks_count,
            "embedding_count": self.embedding_count
        }

    def load_vector_store(self, persist_directory="./chroma_db", collection_name=None):
        """
        Load a persisted vector store from disk
        
        Args:
            persist_directory (str): Directory with the persisted vector store
            collection_name (str): Optional collection name to load
            
        Returns:
            Retriever: Loaded retriever object
        """
        # Use provided collection name or the current one
        coll_name = collection_name if collection_name else self.collection_name
        
        try:
            loaded_db = Chroma(
                collection_name=coll_name,
                embedding_function=self.embedding_function,
                persist_directory=persist_directory
            )
            
            # Update collection name if a specific one was provided
            if collection_name:
                self.collection_name = collection_name
                
            # Get collection stats
            self.chunks_count = loaded_db._collection.count()
            self.embedding_count = self.chunks_count
            
            # Set up retriever
            self.retriever = loaded_db.as_retriever(search_kwargs={"k": 3})
            
            return self.retriever
            
        except Exception as e:
            raise ValueError(f"Error loading vector store: {str(e)}")
    
    async def expand_query(self, query, llm_service=None):
        """
        Expand the user query to improve retrieval
        
        Args:
            query (str): Original user query
            llm_service: LLM service for query expansion
            
        Returns:
            list: List of expanded queries
        """
        # If LLM service not provided, return original query
        if not llm_service:
            return [query]
            
        prompt = f"""
        Given the user query: "{query}"
        Generate 3 alternative versions of this query that might help retrieve relevant information.
        Format your response as a single line with queries separated by semicolons (;).
        """
        
        try:
            expanded_queries_text = await llm_service.agenerate_text(prompt)
            expanded_queries = expanded_queries_text.split(';')
            
            # Add original query and clean up expanded queries
            all_queries = [query] + [q.strip() for q in expanded_queries if q.strip()]
            return all_queries
        except Exception as e:
            # If query expansion fails, return the original query
            print(f"Query expansion error: {str(e)}")
            return [query]
    
    def get_retriever(self):
        """Get the current retriever"""
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Process documents first.")
        return self.retriever
    
    def log_query(self, query, documents, retrieval_time):
        """
        Log query for performance tracking
        
        Args:
            query (str): The search query
            documents (list): Retrieved documents
            retrieval_time (float): Time taken for retrieval
        """
        if not hasattr(self, '_query_history'):
            self._query_history = []
        
        self._query_history.append({
            "query": query,
            "document_count": len(documents),
            "documents": [doc.metadata.get("source", "unknown") for doc in documents],
            "retrieval_time": retrieval_time,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_stats(self):
        """Return statistics about the vector store"""
        return {
            "chunks_count": self.chunks_count,
            "embedding_count": self.embedding_count,
            "collection_name": self.collection_name
        }
    
    def get_advanced_stats(self):
        """
        Get detailed statistics about the vector store and retrieval performance
        
        Returns:
            dict: Detailed statistics and performance metrics
        """
        # Basic stats
        stats = self.get_stats()
        
        # Add performance metrics
        if hasattr(self, '_query_history') and self._query_history:
            stats["total_queries"] = len(self._query_history)
            stats["avg_retrieval_time"] = sum(q.get("retrieval_time", 0) for q in self._query_history) / len(self._query_history)
            stats["avg_documents_retrieved"] = sum(q.get("document_count", 0) for q in self._query_history) / len(self._query_history)
            stats["last_query_time"] = self._query_history[-1].get("timestamp", "unknown")
            
            # Get most frequently retrieved documents
            all_docs = []
            for query in self._query_history:
                all_docs.extend(query.get("documents", []))
            
            if all_docs:
                from collections import Counter
                doc_counts = Counter(all_docs)
                stats["top_documents"] = doc_counts.most_common(5)
        
        return stats
    
    def mmr_search(self, query, k=3, fetch_k=10, lambda_mult=0.7):
        """
        Perform Maximum Marginal Relevance search for diversity
        
        Args:
            query (str): Search query
            k (int): Number of documents to return
            fetch_k (int): Number of documents to consider for diversity
            lambda_mult (float): Balance between relevance and diversity
            
        Returns:
            list: Retrieved documents
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Process documents first.")
            
        start_time = time.time()
        
        docs = Chroma(
            collection_name=self.collection_name,
            client=self.client,
            embedding_function=self.embedding_function
        ).as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        ).get_relevant_documents(query)
        
        retrieval_time = time.time() - start_time
        
        # Log the query for performance tracking
        self.log_query(query, docs, retrieval_time)
        
        return docs
    
    def clear_query_history(self):
        """Clear the query history"""
        self._query_history = []
        return {"status": "Query history cleared"}