import os
import asyncio
from typing import List
from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Import your existing modules
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter

# Load environment variables
load_dotenv()

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Import your existing tiktoken function
import tiktoken
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(text)
    return len(tokens)

# Import your existing read_pdf_documents function
def read_pdf_documents(docs_dir: str = "docs") -> List:
    """
    Read PDF documents from the specified directory and extract their text content.
    """
    documents = []
    
    if not os.path.exists(docs_dir):
        print(f"Directory {docs_dir} does not exist.")
        return documents
    
    for filename in os.listdir(docs_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(docs_dir, filename)
            
            try:
                temp_doc = PyMuPDFLoader(pdf_path).load()
                documents.extend(temp_doc)
                print(f"Loaded doc: {filename}")
                
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
                
    return documents

# Function to initialize your RAG chain
def initialize_cybersecurity_rag_chain():
    """
    Initialize the cybersecurity RAG chain from your existing setup
    """
    # Set up the embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=0,
        length_function=tiktoken_len,
    )
    
    # Load cybersecurity documents
    print("Loading cybersecurity documents...")
    cybersecurity_docs_dir = "docs/cybersecurity"
    
    if os.path.exists(cybersecurity_docs_dir):
        pdf_documents = read_pdf_documents(docs_dir=cybersecurity_docs_dir)
        if pdf_documents:
            cyber_security_split_chunks = text_splitter.split_documents(pdf_documents)
            
            # Create cybersecurity vectorstore
            cyber_security_qdrant_vectorstore = Qdrant.from_documents(
                cyber_security_split_chunks,
                embedding_model,
                location=":memory:",
                collection_name="cybersecurity_requirements",
            )
            cyber_security_qdrant_retriever = cyber_security_qdrant_vectorstore.as_retriever()
            
            # Create cybersecurity RAG prompt and chain
            CYBER_RAG_PROMPT = """
            CONTEXT:
            {context}
    
            QUERY:
            {question}
    
            You are a helpful Cybersecurity Expert in the field of FDA Software as a Medical Device. Use the available context to answer the question. 
            If you can't answer the question, say you don't know.
            """
            cyber_security_rag_prompt = ChatPromptTemplate.from_template(CYBER_RAG_PROMPT)
            cyber_security_openai_chat_model = ChatOpenAI(model="gpt-4o-mini")
            cyber_security_rag_chain = (
                {"context": itemgetter("question") | cyber_security_qdrant_retriever, "question": itemgetter("question")}
                | cyber_security_rag_prompt | cyber_security_openai_chat_model | StrOutputParser()
            )
            
            return cyber_security_rag_chain, cyber_security_qdrant_retriever
    
    return None, None

# Function to load FDA dataset for RAGAS
def load_fda_cybersecurity_dataset_for_ragas():
    """
    Load the FDA Cybersecurity Golden Dataset and transform it for RAGAS evaluation.
    """
    # Load the dataset
    dataset = load_dataset("pratikmurali/FDA_Cybersecurity_Golden_Dataset")
    
    # Get the default split
    if 'train' in dataset:
        data = dataset['train']
    elif 'default' in dataset:
        data = dataset['default']
    else:
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
    
    # Transform to RAGAS format
    def transform_to_ragas(example):
        return {
            'question': example['Question'],
            'ground_truth': example['Answer'],
            'contexts': [example['Reference context']],
            'answer': '',  # Will be filled with generated answers
            'source': example['Source']
        }
    
    ragas_dataset = data.map(transform_to_ragas)
    return ragas_dataset

# Main evaluation function
async def evaluate_cybersecurity_rag_chain():
    """
    Evaluate the cybersecurity RAG chain using RAGAS metrics
    """
    # Initialize the RAG chain
    print("Initializing cybersecurity RAG chain...")
    cyber_security_rag_chain, retriever = initialize_cybersecurity_rag_chain()
    
    if cyber_security_rag_chain is None:
        print("Failed to initialize cybersecurity RAG chain. Please check your documents.")
        return
    
    # Load the FDA dataset
    print("\nLoading FDA Cybersecurity dataset...")
    fda_dataset = load_fda_cybersecurity_dataset_for_ragas()
    print(f"Loaded {len(fda_dataset)} examples")
    
    # Generate answers using the RAG chain
    print("\nGenerating answers with RAG chain...")
    generated_answers = []
    questions = fda_dataset['question']
    
    # Use tqdm for progress bar
    for question in tqdm(questions, desc="Generating answers", unit="question"):
        try:
            # Invoke your RAG chain
            answer = cyber_security_rag_chain.invoke({"question": question})
            generated_answers.append(answer)
                
        except Exception as e:
            print(f"\nError generating answer: {str(e)}")
            generated_answers.append("Error generating answer")
    
    # Prepare evaluation dataset
    print("\nPreparing dataset for evaluation...")
    df = fda_dataset.to_pandas()
    df['answer'] = generated_answers
    
    # Get the contexts used by the retriever for each question
    print("\nRetrieving contexts used by the RAG chain...")
    retrieved_contexts = []
    
    # Use tqdm for progress bar
    for question in tqdm(questions, desc="Retrieving contexts", unit="question"):
        try:
            docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in docs]
            retrieved_contexts.append(contexts)
        except Exception as e:
            print(f"\nError retrieving contexts: {str(e)}")
            retrieved_contexts.append([])
    
    # Update the contexts with what was actually retrieved
    df['contexts'] = retrieved_contexts
    
    # Convert back to Dataset
    eval_dataset = Dataset.from_pandas(df)
    
    # Run RAGAS evaluation
    print("\nRunning RAGAS evaluation...")
    
    # Show progress during evaluation
    metrics_to_evaluate = [
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity
    ]
    
    results = evaluate(
        eval_dataset,
        metrics=metrics_to_evaluate
    )
    
    # Print results
    print("\n=== Evaluation Results ===")
    
    # Convert results to dictionary format
    if hasattr(results, 'to_pandas'):
        # If results has to_pandas method
        results_df = results.to_pandas()
        results_dict = results_df.to_dict('records')[0] if len(results_df) > 0 else {}
    elif hasattr(results, '__dict__'):
        # If results is an object with attributes
        results_dict = results.__dict__
    else:
        # If results is already a dictionary
        results_dict = results
    
    # Print each metric
    for metric, score in results_dict.items():
        if isinstance(score, (int, float)):
            print(f"{metric}: {score:.4f}")
    
    # Save detailed results
    results_df = pd.DataFrame([results_dict])
    results_df.to_csv("fda_cybersecurity_ragas_results.csv", index=False)
    
    # Save detailed evaluation data
    eval_df = eval_dataset.to_pandas()
    eval_df.to_csv("fda_cybersecurity_detailed_evaluation.csv", index=False)
    
    print("\nResults saved to:")
    print("- fda_cybersecurity_ragas_results.csv (metrics summary)")
    print("- fda_cybersecurity_detailed_evaluation.csv (full evaluation data)")
    
    # Create visualizations
    create_evaluation_visualizations(results_dict, eval_df)
    
    return results_dict

# Function to run a single example for testing
def test_single_example():
    """
    Test the RAG chain with a single example from the dataset
    """
    # Initialize the RAG chain
    print("Initializing cybersecurity RAG chain...")
    cyber_security_rag_chain, retriever = initialize_cybersecurity_rag_chain()
    
    if cyber_security_rag_chain is None:
        print("Failed to initialize cybersecurity RAG chain.")
        return
    
    # Load the FDA dataset
    dataset = load_fda_cybersecurity_dataset_for_ragas()
    
    # Test with the first question
    first_example = dataset[0]
    question = first_example['question']
    
    print(f"\nQuestion: {question}")
    print(f"Ground Truth: {first_example['ground_truth']}")
    
    # Get answer from RAG chain
    answer = cyber_security_rag_chain.invoke({"question": question})
    print(f"\nGenerated Answer: {answer}")
    
    # Get retrieved contexts
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    print(f"\nRetrieved Contexts: {contexts[0][:200]}..." if contexts else "No contexts retrieved")

def create_evaluation_visualizations(results_dict, eval_df):
    """
    Create visualizations for the RAGAS evaluation results
    """
    print("\nCreating visualizations...")
    
    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter out non-numeric values from results
    numeric_results = {k: v for k, v in results_dict.items() if isinstance(v, (int, float))}
    
    if not numeric_results:
        print("No numeric results found for visualization")
        return
    
    # 1. Bar chart of all metrics
    plt.figure(figsize=(10, 6))
    metrics = list(numeric_results.keys())
    scores = list(numeric_results.values())
    
    bars = plt.bar(metrics, scores, color=sns.color_palette("husl", len(metrics)))
    
    # Add value labels on bars
    for i, (metric, score) in enumerate(zip(metrics, scores)):
        plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('RAGAS Evaluation Metrics', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'ragas_metrics_bar_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Radar chart for metrics
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Prepare data for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    scores_radar = scores + scores[:1]  # Complete the circle
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, scores_radar, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, scores_radar, alpha=0.25, color='#1f77b4')
    
    # Customize the plot
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=12)
    ax.set_title('RAGAS Metrics Radar Chart', size=16, fontweight='bold', pad=20)
    ax.grid(True)
    
    # Add score labels
    for angle, score, metric in zip(angles[:-1], scores, metrics):
        ax.text(angle, score + 0.05, f'{score:.3f}', ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'ragas_metrics_radar_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Horizontal bar chart with color coding
    plt.figure(figsize=(10, 8))
    
    # Sort metrics by score
    sorted_metrics = sorted(zip(metrics, scores), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_scores = zip(*sorted_metrics)
    
    # Create color map based on score
    colors = ['green' if score >= 0.8 else 'orange' if score >= 0.6 else 'red' for score in sorted_scores]
    
    bars = plt.barh(sorted_names, sorted_scores, color=colors)
    
    # Add value labels
    for i, (name, score) in enumerate(zip(sorted_names, sorted_scores)):
        plt.text(score + 0.01, i, f'{score:.3f}', va='center', fontweight='bold')
    
    plt.xlabel('Score', fontsize=12)
    plt.title('RAGAS Metrics Performance (Sorted)', fontsize=16, fontweight='bold')
    plt.xlim(0, 1.1)
    plt.grid(axis='x', alpha=0.3)
    
    # Add performance level indicators
    plt.axvline(x=0.8, color='green', linestyle='--', alpha=0.7, label='Good (≥0.8)')
    plt.axvline(x=0.6, color='orange', linestyle='--', alpha=0.7, label='Fair (≥0.6)')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'ragas_metrics_horizontal_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distribution of scores per question (if individual scores available)
    if 'question' in eval_df.columns and len(eval_df) > 1:
        plt.figure(figsize=(12, 6))
        
        # Create a simple line plot showing performance across questions
        question_numbers = range(1, len(eval_df) + 1)
        
        # Plot answer correctness if available
        if 'answer_correctness' in eval_df.columns:
            plt.plot(question_numbers, eval_df['answer_correctness'], 
                    marker='o', label='Answer Correctness', linewidth=2)
        
        # Plot answer similarity if available
        if 'answer_similarity' in eval_df.columns:
            plt.plot(question_numbers, eval_df['answer_similarity'], 
                    marker='s', label='Answer Similarity', linewidth=2)
        
        plt.xlabel('Question Number', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Performance Across Questions', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f'ragas_question_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Create a summary report image
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FDA Cybersecurity RAG Evaluation Summary', fontsize=20, fontweight='bold')
    
    # Metric scores (top-left)
    ax1.axis('tight')
    ax1.axis('off')
    table_data = [[metric, f'{score:.3f}'] for metric, score in numeric_results.items()]
    table = ax1.table(cellText=table_data, colLabels=['Metric', 'Score'], 
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax1.set_title('Metric Scores', fontsize=14, fontweight='bold')
    
    # Mini bar chart (top-right)
    ax2.bar(range(len(metrics)), scores, color=sns.color_palette("husl", len(metrics)))
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels([m[:8] for m in metrics], rotation=45, ha='right')
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Metric Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Average performance (bottom-left)
    avg_score = np.mean(scores)
    ax3.pie([avg_score, 1-avg_score], labels=[f'Score\n{avg_score:.3f}', ''], 
            colors=['#2ecc71', '#ecf0f1'], startangle=90, counterclock=False)
    ax3.set_title('Average Performance', fontsize=14, fontweight='bold')
    
    # Key insights (bottom-right)
    ax4.axis('off')
    best_metric = max(numeric_results.items(), key=lambda x: x[1])
    worst_metric = min(numeric_results.items(), key=lambda x: x[1])
    
    insights = [
        f"Best performing metric: {best_metric[0]} ({best_metric[1]:.3f})",
        f"Lowest performing metric: {worst_metric[0]} ({worst_metric[1]:.3f})",
        f"Average score: {avg_score:.3f}",
        f"Metrics above 0.8: {sum(1 for s in scores if s >= 0.8)}/{len(scores)}",
        f"Total questions evaluated: {len(eval_df)}"
    ]
    
    for i, insight in enumerate(insights):
        ax4.text(0.05, 0.8 - i*0.15, f"• {insight}", transform=ax4.transAxes, 
                fontsize=12, va='center')
    ax4.set_title('Key Insights', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'ragas_evaluation_summary_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved with timestamp: {timestamp}")
    print(f"- ragas_metrics_bar_{timestamp}.png")
    print(f"- ragas_metrics_radar_{timestamp}.png")
    print(f"- ragas_metrics_horizontal_{timestamp}.png")
    print(f"- ragas_question_performance_{timestamp}.png")
    print(f"- ragas_evaluation_summary_{timestamp}.png")

if __name__ == "__main__":
    # Test with a single example first
    print("Testing with a single example...")
    test_single_example()
    
    # Run full evaluation
    print("\n\nRunning full evaluation...")
    results = asyncio.run(evaluate_cybersecurity_rag_chain())