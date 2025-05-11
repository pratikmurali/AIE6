# SAMD Regulatory Assistance Bot

## Overview
This chatbot provides assistance with FDA regulatory topics related to Software as a Medical Device (SaMD). It leverages two specialized knowledge domains:

1. **FDA Regulatory Information**: Guidelines on AI/ML topics for SaMD, premarket software functions, quality management systems, and reporting gaps.

2. **Cybersecurity Requirements**: Information on cybersecurity requirements for FDA submissions, pre/post-market requirements, NIST framework, and related legislation.

## How to Use
Simply ask questions about FDA regulations for medical device software or cybersecurity requirements for medical devices. The bot will automatically:

1. Route your question to the appropriate expert system
2. Research the relevant information
3. Provide a comprehensive answer

## Example Questions
- "What are the components of a 510(k) submission for software as a medical device?"
- "Under what circumstances is a PCCP required?"
- "Explain the QMS requirements for SAMD. How are cybersecurity requirements handled in QMS?"
- "What are the cybersecurity requirements for FDA-approved AI medical devices?"

This application utilizes LangGraph for orchestrating a multi-agent system and RAG (Retrieval Augmented Generation) for accessing specialized knowledge. 