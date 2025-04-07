import os
from typing import List
import PyPDF2
import sys


class PDFLoader:
    def __init__(self, path: str):
        """
        Initialize a PDF loader that reads content from PDF files.
        
        Parameters
        ----------
        path : str
            Path to a PDF file or a directory containing PDF files.
        """
        self.documents = []
        self.path = path
        
    def load(self):
        """
        Load PDF documents from the specified path.
        
        If path is a directory, all PDF files in the directory will be loaded.
        If path is a file, only that file will be loaded if it's a PDF.
        """
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.lower().endswith(".pdf"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a PDF file."
            )
    
    def load_file(self):
        """Load a single PDF file and extract its text content."""
        with open(self.path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            self.documents.append(text)
    
    def load_directory(self):
        """Load all PDF files from a directory and its subdirectories."""
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        self.documents.append(text)
    
    def load_documents(self):
        """
        Load and return all PDF documents.
        
        Returns
        -------
        List[str]
            List of text content extracted from PDF documents.
        """
        self.load()
        return self.documents


class CharacterPDFSplitter:
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
    ):
        """
        Initialize a text splitter with specified chunk size and overlap.
        
        Parameters
        ----------
        chunk_size : int, default=1024
            The size of each chunk in characters.
        chunk_overlap : int, default=200
            The number of characters of overlap between chunks.
        """
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        """
        Split a single text into chunks with specified overlap.
        
        Parameters
        ----------
        text : str
            The text to split.
            
        Returns
        -------
        List[str]
            List of text chunks.
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        """
        Split multiple texts into chunks with specified overlap.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to split.
            
        Returns
        -------
        List[str]
            List of all text chunks from all input texts.
        """
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    # Example usage
    # Use command line argument for PDF path if provided, otherwise use a default
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "example.pdf"
    
    print(f"Attempting to load PDF from: {pdf_path}")
    loader = PDFLoader(pdf_path)
    
    try:
        documents = loader.load_documents()
        splitter = CharacterPDFSplitter(chunk_size=1024, chunk_overlap=200)
        chunks = splitter.split_texts(documents)
        
        # Print some example chunks
        if chunks:
            print(f"Total chunks: {len(chunks)}")
            print("\nFirst chunk:")
            print(chunks[0][:100] + "...")
            if len(chunks) > 1:
                print("\nSecond chunk (notice the overlap):")
                print(chunks[1][:100] + "...")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nUsage: python pdf_utils.py [path_to_pdf]")
        print("Please provide a valid path to a PDF file.")

