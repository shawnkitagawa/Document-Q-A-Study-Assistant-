# Culinary Technique RAG System

A modular culinary intelligence system that combines **LLM-based intent normalization**, **retrieval-augmented generation (RAG)**, and **decision-based technique planning** to answer cooking questions and generate grounded cooking strategy.

This project is designed to go beyond simple recipe generation.  
It separates the system into clear layers:

- **Intent understanding**
- **Decision planning**
- **Evidence retrieval**
- **Grounded culinary reasoning**
- **Technique plan storage**
- **API access through FastAPI**

---

## Project Goal

The goal of this project is to build a cooking assistant that can:

- answer culinary questions from a book-based knowledge base
- ingest and index cooking PDFs into a vector database
- normalize vague user requests into structured cooking intent
- generate a **technique plan** instead of blindly producing recipes
- support future expansion into full recipe generation
- prepare for easy cloud deployment on **Google Cloud Platform (GCP)**

---

## Core Idea

This system separates responsibilities clearly:

### 1. LLM-only normalization layer
The LLM is responsible for understanding the user request:

- Is this a recipe-related request?
- What ingredient, cuisine, flavor, or intent is present?
- Are there equipment, dietary, or time constraints?
- Is the request clear enough to execute?

It does **not** generate the recipe here.  
It only converts messy natural language into structured variables.

### 2. Deterministic decision layer
Once the request is normalized, the system activates a set of relevant **decision keys** such as:

- method selection
- doneness and heat control
- sauce architecture
- acid strategy
- salt strategy
- plating and finish
- failure prevention

This step is deterministic and policy-based.  
It does not rely on hallucinated planning.

### 3. RAG evidence retrieval
For each decision key, the system generates retrieval queries and searches the Chroma vector database built from culinary PDFs.

This gives grounded evidence from books and cooking references.

### 4. Technique Plan generation
The LLM then receives:

- the decision key
- the normalized request
- stable decision questions
- retrieved evidence chunks

It produces a structured **Technique Plan JSON** with:

- a decision
- reasoning
- confidence
- citations
- whether more evidence is needed

---

## Features

- PDF ingestion and chunking
- ChromaDB persistent vector storage
- OpenAI embeddings
- document-based culinary Q&A
- structured request normalization
- deterministic decision-key planning
- evidence-backed technique plan generation
- FastAPI endpoints for upload, question answering, and recipe planning
- future-ready GCP deployment structure

---

## Project Structure

```text
project/
│
├── app/
│   └── app.py                 # FastAPI entry point
│
├── recipe/
│   ├── assemble.py            # Main recipe planning pipeline
│   ├── normalize.py           # LLM intent normalization
│   ├── decision_keys.py       # Deterministic decision activation + RAG query generation
│   ├── technique.py           # Technique Plan generation with LLM + RAG
│   └── ...                    # Future recipe assembly modules
│
├── data/
│   ├── ingredients/           # Ingredient datasets
│   ├── raw/                   # Uploaded PDF files
│   ├── vectorstore/           # Persistent ChromaDB storage
│   └── infrastructure/        # Deployment / infrastructure-related files
│
├── config.py                  # Global config, model setup, collection config
├── ingest.py                  # PDF extraction, chunking, and Chroma ingestion
├── rag.py                     # Retrieval and grounded answer generation
├── utils.py                   # Cleaning and chunking helpers
│
└── technique_plans/           # Saved generated technique plan JSON files