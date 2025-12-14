Document Q&A Study Assistant (RAG System)

A Retrieval-Augmented Generation (RAG) system that ingests PDF lecture notes, retrieves relevant information using vector embeddings, and generates grounded answers using a Large Language Model (LLM).

This project focuses on end-to-end RAG system design, modular architecture, and preparation for cloud deployment and advanced retrieval strategies.



# Features

PDF ingestion and preprocessing

Semantic chunking and vector embedding

Vector database storage using ChromaDB

Top-K similarity-based retrieval

Context-grounded LLM responses

Explicit handling of out-of-scope questions ("I don't know")

Source citation for transparency

Modular, extensible architecture




# Project Architecture
Document-Q-A-Study-Assistant/
│
├── config.py        # Configuration and setup
├── utils.py         # Helper functions
├── ingest.py        # Vector database construction
├── rag.py           # Core RAG logic
├── app.py           # Application entry point (UI / runner)
└── lecture_notes/   # Input PDF documents

# File Overview

config.py

Centralized configuration file:

Embedding function setup

ChromaDB collection initialization

LLM model selection

Secure API key loading from .env




## utils.py

Reusable helper functions for:

Text processing

Chunking

Utility logic shared across modules



## ingest.py  

Responsible for building the vector database:

Load PDF files

Extract raw text

Split text into chunks

Convert chunks into embeddings

Store embeddings in ChromaDB

This step is run once or whenever new documents are added.




## rag.py

Core Retrieval-Augmented Generation logic:

Retrieve relevant chunks from ChromaDB

Build a structured context for the LLM

Generate an answer using the LLM

Return the answer along with source references




## app.py

Application entry point and user interface:

Accepts user questions

Calls rag.answer_question()

Prints the final answer with cited sources



# Future Plans
AWS Deployment

Deploy the RAG system as a cloud-hosted API:

EC2 / ECS for backend server

S3 for persistent PDF storage

Secrets Manager for API keys

CloudWatch for logging and monitoring

Optional Cognito for user authentication

This will allow:

Always-on access

No reliance on local computing power

Multi-user scalability





## Strategy 1: Reranking (Two-Stage Retrieval)

Add an optional reranking step:

Retrieve top-N chunks using vector similarity

Reorder them using a more precise reranker model

Select the best K chunks for generation

Purpose:

Improve answer precision

Reduce irrelevant context

Enable speed vs accuracy tradeoff

Planned as a configurable toggle (A/B comparison).



## Strategy 2: Graph Retrieval

Extend retrieval using relationship-based search:

Build a knowledge graph of entities (authors, topics, references)

Traverse relationships to answer multi-hop questions

Combine with vector retrieval for a hybrid RAG system

Use cases:

Author attribution

Citation relationships

Structured factual queries



# Planned Enhancements

Retrieval quality evaluation and logging

Similarity thresholding for safer “I don’t know” responses

Rate limiting and cost control

Simple web UI for demonstrations
