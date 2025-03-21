from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings
from chromadb import Client
from langchain.vectorstores import Chroma
import uuid

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

    def setup_chroma(self, documents):
        """
        Set up Chroma with documents
        
        Args:
            documents (list): List of Document objects
            
        Returns:
            Retriever: Chroma retriever object
        """
        # Delete existing collection if present
        existing_collections = self.client.list_collections()
        collection_names = [col.name for col in existing_collections]
        if self.collection_name in collection_names:
            self.client.delete_collection(self.collection_name)

        collection = self.client.create_collection(self.collection_name)

        # Batch process embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_function.embed_documents(texts)

        # Add documents with metadata
        for idx, (text, doc) in enumerate(zip(texts, documents)):
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
    
    def get_retriever(self):
        """Get the current retriever"""
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Process documents first.")
        return self.retriever