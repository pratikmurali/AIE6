# SAMD Regulatory Assistance Bot

A Chainlit application that provides FDA regulatory information and cybersecurity guidance for Software as a Medical Device (SaMD).

## Features

- Answers questions about FDA regulatory topics related to software as a medical device
- Provides cybersecurity guidance for medical device software
- Utilizes RAG (Retrieval Augmented Generation) with specialized knowledge bases
- Implements a multi-agent system with LangGraph
- Streams responses in real-time

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

4. Create the necessary folders and add PDF documents:
```bash
mkdir -p docs docs/cybersecurity
```

5. Add your FDA regulatory PDF documents to the `docs` folder and cybersecurity documents to the `docs/cybersecurity` folder.

## Usage

1. Run the Chainlit application:
```bash
chainlit run main.py
```

2. Open your browser and go to http://localhost:8000

3. Ask questions about FDA regulations for medical device software or cybersecurity requirements.

## Example Questions

- "What are the components of a 510(k) submission for software as a medical device?"
- "Under what circumstances is a PCCP required?"
- "Explain the QMS requirements for SAMD. How are cybersecurity requirements handled in QMS?"
- "What are the cybersecurity requirements for FDA-approved AI medical devices?"

## Project Structure

- `main.py`: The main Chainlit application
- `docs/`: Folder containing FDA regulatory PDF documents
- `docs/cybersecurity/`: Folder containing cybersecurity PDF documents
- `requirements.txt`: List of required packages
- `chainlit.md`: Introduction information displayed in the Chainlit UI

## Gold Test Datasets
- Gold Test Dataset for Cybersecurity RAG : https://huggingface.co/datasets/pratikmurali/FDA_Cybersecurity_Golden_Dataset
- Gold Test Dataset for FDA Regulatory Submission for SAMD RAG : https://huggingface.co/datasets/pratikmurali/fda_samd_regulations_golden_test_dataset
