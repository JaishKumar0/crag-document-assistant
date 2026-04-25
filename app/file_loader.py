from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)

def load_document(path: str):
    if path.endswith(".pdf"):
        return PyPDFLoader(path).load()
    elif path.endswith(".txt"):
        return TextLoader(path).load()
    elif path.endswith(".docx"):
        return UnstructuredWordDocumentLoader(path).load()
    else:
        raise ValueError("Unsupported format")