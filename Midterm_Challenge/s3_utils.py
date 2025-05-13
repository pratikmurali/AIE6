import os
import boto3
import tempfile
import botocore
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from botocore.config import Config

def download_and_read_s3_documents(bucket_name: str = "fda-samd-cybersecurity-guidance") -> List:
    """
    Download PDF documents from the specified S3 bucket and extract their text content.
    
    Args:
        bucket_name (str): S3 bucket name containing PDF documents
        
    Returns:
        List: List of extracted text content from PDFs
    """
    documents = []
    
    try:
        # Configure anonymous access
        s3_config = Config(
            signature_version=botocore.UNSIGNED,
            region_name='us-east-1'
        )
        
        # Create S3 client with anonymous access configuration
        s3_client = boto3.client(
            's3',
            config=s3_config,
        )
        
        # Create temporary directory to store downloaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Created temporary directory: {temp_dir}")
            
            # List objects in the S3 bucket
            try:
                response = s3_client.list_objects_v2(Bucket=bucket_name)
            except botocore.exceptions.ClientError as e:
                print(f"Error accessing S3 bucket: {e}")
                return documents
                
            if 'Contents' not in response:
                print("No contents found in the S3 bucket.")
                return documents
                
            # Download each PDF file and process it
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.pdf'):
                    local_file_path = os.path.join(temp_dir, os.path.basename(key))
                    
                    try:
                        print(f"Downloading {key} from S3...")
                        s3_client.download_file(bucket_name, key, local_file_path)
                        
                        temp_doc = PyMuPDFLoader(local_file_path).load()
                        documents.extend(temp_doc)
                        print(f"Loaded doc: {key}")
                    except Exception as e:
                        print(f"Error processing {key}: {str(e)}")
    
    except Exception as e:
        print(f"Error in download_and_read_s3_documents: {str(e)}")
        
    return documents 