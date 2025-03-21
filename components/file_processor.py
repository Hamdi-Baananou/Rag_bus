import os
import re
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader

def process_pdfs(pdf_files):
    """
    Process PDFs with error handling and text cleaning
    
    Args:
        pdf_files (list): List of file paths to PDF files
        
    Returns:
        list: List of Document objects
    """
    all_documents = []

    if not pdf_files:
        raise ValueError("No PDF files provided")

    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_file)
            documents = loader.load()

            # Clean and combine text
            full_text = "\n".join([
                re.sub(r'\s+', ' ', doc.page_content).strip()
                for doc in documents
            ])

            full_doc = Document(
                page_content=full_text,
                metadata={"source": os.path.basename(pdf_file)}
            )
            all_documents.append(full_doc)
            print(f"Processed {pdf_file} successfully")

        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue

    return all_documents