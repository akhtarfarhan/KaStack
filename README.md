]# 🛡️ AI/ML Message Processing Pipeline
**KaStack Labs - AI/ML Engineer Intern Assignment**

## 🚀 Overview
This repository contains a privacy-first, local Natural Language Processing (NLP) pipeline designed to ingest, classify, and extract structured information from internal communications. 

**The primary architectural constraint:** Absolute data privacy. To comply with strict internal security protocols, **no raw data is sent to external cloud LLM APIs.** The entire pipeline utilizes local rule-based heuristics and local Hugging Face transformer models to ensure zero data leakage.

## ✨ Key Features
1. **Privacy Firewall (Zero-Leakage):** Scans and masks highly sensitive PII and financial data using strict Regex patterns *before* any ML processing occurs.
2. **Local Zero-Shot Classification:** Categorizes messages using an on-device `facebook/bart-large-mnli` transformer model.
3. **Deterministic Entity Extraction:** Utilizes `spaCy` (en_core_web_sm) combined with temporal logic to accurately extract tasks, events, dates, and assignees without AI hallucination.
4. **Interactive Dashboard:** A Streamlit-powered UI to visualize classifications, extractions, and masked data.

---

## 🏗️ System Architecture & Workflow

### Phase 1: Sensitive Information Detection & Masking (`src/masker.py`)
To ensure safety, we employ a "Privacy Firewall". 
* **Mechanism:** Uses Regex to detect standardized sensitive formats (16-digit credit cards, OTPs, Authentication Tokens, Passwords).
* **Action:** Replaces identified strings with `******` and assigns a risk severity (High/Medium).
* **Result:** Generates `outputs/masked_messages.csv`, a sanitized dataset used exclusively for all downstream tasks.

### Phase 2: Message Classification (`src/classifier.py`)
* **Mechanism:** The sanitized text is passed through a local Hugging Face zero-shot classifier.
* **Fast-Path Heuristic:** If a message contains the `******` mask, the system bypasses the heavy NLP model and instantly categorizes it as "Sensitive Information" (Confidence: 0.99).
* **Reasoning Engine:** A rule-based mapper generates logical reasoning strings based on the model's predicted category output.

### Phase 3: Task & Event Extraction (`src/extractor.py`)
* **Mechanism:** Filters the dataset for "Action Required" and "Meeting or Event" categories, passing them through `spaCy` for Named Entity Recognition (NER).
* **Temporal Grounding:** Resolves relative time constraints (e.g., "tomorrow") by anchoring them to the absolute timestamp of the message row.
* **Anti-Hallucination:** If an entity (time, date, person) is not explicitly detected, it is strictly stored as `null`.

---

## 📂 Project Structure
```text
kastack_ml_assignment/
├── data/                       # Raw datasets (Git-ignored for privacy)
├── outputs/                    # Generated JSON reports and masked CSVs
├── src/
│   ├── masker.py               # Phase 1: Privacy Firewall
│   ├── classifier.py           # Phase 2: Zero-Shot Classification
│   └── extractor.py            # Phase 3: SpaCy Entity Extraction
├── app.py                      # Streamlit Dashboard UI
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation