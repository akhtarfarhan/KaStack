# AI/ML Message Processing Pipeline (L1 & L2 Architecture)
**KaStack Labs - AI/ML Engineer Intern Assignment Submission**

## Overview
This repository contains a privacy-first, local Natural Language Processing (NLP) pipeline designed to ingest, classify, semantically group, and extract structured information from internal communications. 

**The Prime Constraint:** Absolute data privacy. To comply with strict internal security protocols, **no raw data is sent to external cloud LLM APIs.** The entire pipeline utilizes local rule-based heuristics and local Hugging Face transformer models to ensure zero data leakage.

---

## System Evolution: How L2 Extends L1
The **L1 System** established a deterministic, isolated processing pipeline. It successfully masked sensitive data, classified messages using zero-shot inference, and extracted tasks using standard NER. However, it processed messages in a vacuum without historical context.

The **L2 System** introduces **State Management, Semantic Context, and Intelligent Retrieval (RAG)**. Instead of treating messages in isolation, the L2 architecture links them chronologically and semantically, allowing the system to track the complete lifecycle of a task from creation to completion, dynamically adjusting priorities and statuses along the way.

---

## Core Architectures & Methodologies

### 1. Privacy-Aware Routing & Masking (Phase 1)
To ensure zero leakage, the system employs a deterministic "Privacy Firewall" before any ML processing occurs.
* **Mechanism:** Uses strict Regex to detect standardized sensitive formats (16-digit credit cards, OTPs, Authentication Tokens, Passwords).
* **Routing:** Replaces identified strings with `******` and assigns a risk severity. It dynamically routes items for human confirmation (`ask_for_confirmation`) or completely blocks external processing (`do_not_send_to_external_service`).
* **Result:** Generates a sanitized dataset used exclusively for all downstream ML tasks.

### 2. Message Classification & Entity Extraction (L1 Foundation)
* **Classification:** Sanitized text passes through a local zero-shot classifier (`facebook/bart-large-mnli`). It uses a fast-path heuristic to instantly flag masked strings as "Sensitive" (0.99 confidence), saving compute.
* **Extraction:** Uses `spaCy` (`en_core_web_sm`) coupled with temporal grounding to extract tasks and events. 
* **Anti-Hallucination:** Relative dates (e.g., "tomorrow") are resolved using the exact message timestamp. Missing parameters are strictly stored as `null`.

### 3. Related-Message Grouping Engine (L2 Extension)
* **Mechanism:** Converts messages into dense vector representations using a lightweight, local embedding model (`all-MiniLM-L6-v2` via `SentenceTransformers`).
* **Identification:** Calculates **cosine similarity** between chronologically sorted messages. If the similarity score exceeds the confidence threshold (0.65), messages are grouped into continuous threads.
* **Status Updates:** The engine evaluates the chronological flow of the thread to determine the final status (e.g., a "Pending" task becomes "Completed" or "Rescheduled" based on subsequent follow-ups).

### 4. Dynamic Priority Engine (L2 Extension)
* **Mechanism:** Calculates priority dynamically by evaluating the entire semantic group rather than isolated messages.
* **Logic:** Analyzes the chronological context for signals like `imminent_deadline`, `urgent_follow_up`, or `status_cancelled`.
* **Updates:** If an initial message requests a report (High Priority), but a follow-up says "The report is cancelled", the engine dynamically updates the final group priority to Low.

### 5. Semantic Search & Intelligent Assistant (RAG)
* **Mechanism:** A localized Retrieval-Augmented Generation (RAG) assistant. 
* **Retrieval:** Converts user queries into vector embeddings and performs a semantic search against the pre-computed L2 message groups.
* **Routing:** Uses intent-matching to route specific administrative queries (e.g., "Which tasks are blocked?") directly to the Privacy or Priority dataframes for instant, deterministic $O(1)$ lookups.

---

## Optimization & Benchmarking (L1 vs L2)
*Benchmarking was performed locally on a standard CPU environment.*

| Metric | L1 Architecture | L2 Architecture (Optimized) |
|--------|----------------|----------------|
| **Data Processed** | 900 isolated messages | 1104 chronologically linked messages |
| **Model Size** | 1.6 GB (`bart-large-mnli`) | 1.6 GB + 90 MB (`all-MiniLM`) |
| **Processing Time** | ~3-4 Minutes | ~4-5 Minutes (Minimal embedding latency) |
| **Result Quality** | Static rules. | Dynamic state. Statuses and priorities update contextually. |

**Key L2 Optimization:** To keep the Intelligent Assistant's response times near-instantaneous during live deployment, the L2 system **pre-computes semantic group embeddings** into memory. When a user asks a question, the assistant only performs a lightning-fast cosine similarity matrix multiplication, rather than passing data back through a heavy transformer network.

---

## Assumptions & Limitations
* **Assumptions:** Timestamps are assumed to be in a uniform standard timezone (UTC). Priority relies heavily on explicit keywords (e.g., "urgent", "deadline") and timeline proximity.
* **Limitations:** CPU-based inference for `bart-large-mnli` is functional but slow for massive datasets. Zero-shot models occasionally struggle with multi-label edge cases. Semantic similarity grouping can sometimes falsely link separate tasks if they use highly identical vocabulary.
* **Future Improvements:** In production, this pipeline would benefit from fine-tuning a quantized DistilBERT for faster classification inference, and utilizing a graph-database (like Neo4j) to map message relationships more reliably than pure cosine similarity.

## AI-Tool Usage Declaration
In accordance with assignment rules, Large Language Models were used strictly as an interactive pair-programmer to assist with architectural planning, regex pattern generation, and Streamlit dependency debugging. **No assignment data was uploaded to external LLMs**, and all execution, extraction logic, pipeline orchestration, and local NLP model selections were manually engineered and verified on my local machine.

---
## Repository Structure
```text
kastack_ml_assignment/
├── data/                       # Raw L1 & L2 datasets (Git-ignored for privacy)
├── outputs/                    # Generated JSON structured outputs
├── src/
│   ├── data_merger.py          # Merges and sorts L1/L2 chronologically
│   ├── masker.py               # Privacy Firewall (Phase 1)
│   ├── classifier.py           # Zero-Shot Classification
│   ├── extractor.py            # SpaCy Entity Extraction
│   ├── grouper.py              # Semantic Grouping Engine (L2)
│   ├── priority_engine.py      # Dynamic Priority Logic (L2)
│   └── assistant.py            # RAG Search Assistant (L2)
├── app.py                      # Streamlit Dashboard UI
├── requirements.txt            # Python dependencies (pandas, streamlit)
└── README.md                   # System documentation