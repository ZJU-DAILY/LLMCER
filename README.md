
# 🚀 LLMCER

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green?style=for-the-badge&logo=openai)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **A High-Performance, Modular, and Robust Framework for Entity Resolution using Large Language Models.**

---

## ✨ Key Features

*   **⚡ Parallel Execution**: Optimized with `ThreadPoolExecutor` for concurrent processing of entity blocks, significantly reducing runtime.
*   **🛡️ Misclustering Detection Guardrail (MDG)**: A novel safety mechanism that automatically detects and rejects hallucinatory or illogical clustering outputs from the LLM.
*   **🧩 Modular Architecture**: Clean separation of concerns (Vectorization, Clustering, LLM Interaction, Metrics) for easy maintenance and extensibility.
*   **🔍 Two-Stage Resolution**:
    1.  **Separation**: Splits over-clustered blocks into finer groups.
    2.  **Merging**: Re-evaluates and merges similar groups to ensure high recall.
*   **📊 Comprehensive Metrics**: Automatic calculation of **Purity**, **Inverse Purity**, **F-Measure**, and **ARI** (Adjusted Rand Index).
*   **📝 Smart Logging**: Real-time logging with timestamped files and detailed token usage statistics.

---

## 📂 Project Structure

The project is organized for clarity and scalability:

```plaintext
LLMCER/
├── llmcer/                 # 🧠 Core Library Code
│   ├── clustering.py       # K-Means, LSH, and MDG logic
│   ├── config.py           # Centralized configuration
│   ├── data_utils.py       # Data loading and prompt generation
│   ├── llm_interaction.py  # OpenAI API handling & prompt engineering
│   ├── metrics.py          # Evaluation metrics (F1, ARI, etc.)
│   ├── pipeline.py         # Main orchestration logic (Separation & Merge)
│   ├── vectorization.py    # Embedding generation (Sentence-BERT)
│   └── ...
├── scripts/                # 🏃‍♂️ Execution Scripts
│   └── run_pipeline.py     # Entry point for the python pipeline
├── datasets/               # 💾 Data Storage
│   └── demo_dataset/       # Example datasets
├── logs/                   # 📝 Runtime Logs (Auto-generated)
├── run.sh                  # 🚀 Master Execution Script (Config & Run)
└── requirements.txt        # 📦 Dependencies
```

---

## 📊 Datasets

The repository includes several benchmark datasets for Entity Resolution:

*   **Cora**: Citation data (Author, Title, Venue, Year).
*   **Citesheer**: Scientific publications and citations.
*   **Google-DBLP**: Bibliographic data from Google Scholar and DBLP.
*   **Walmart-Amazon**: Product matching between Walmart and Amazon.
*   **Music20K**: Large-scale music records.
*   **Sigmod**: Database records from SIGMOD.
*   **Affiliation**: Academic affiliation strings.
*   **Song**: Song metadata.

---

## 🚀 Getting Started

### 1. Prerequisites

*   **Python 3.8+**
*   An active **OpenAI API Key**

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/LLMCER.git
cd LLMCER
pip install -r requirements.txt
```

### 3. Configuration

The project is controlled via the `run.sh` script. You **must** configure your environment variables here before running.

Open `run.sh` and update the following:

```bash
# ==========================================
# Configuration Section
# ==========================================

# 1. OpenAI API Key
export OPENAI_API_KEY="sk-..."  # <--- Put your actual API Key here

# 2. Dataset Paths
export DATASET_PATH="/abs/path/to/your/dataset.xlsx"
export GROUND_TRUTH_PATH="/abs/path/to/your/ground_truth.txt"

# 3. Model Configuration
# Path to local embedding model or Hugging Face model name
export EMBEDDING_MODEL_PATH="./all-MiniLM-L6-v2" 
export OPENAI_MODEL="gpt-4o-mini"
```

---

## 🖥️ Usage

Once configured, simply execute the shell script to start the pipeline:

```bash
./run.sh
```

### What happens next?

1.  **Vectorization**: The dataset is converted into vector embeddings.
2.  **Blocking**: LSH groups similar records into coarse blocks.
3.  **Separation**: LLM analyzes blocks and splits them into pure clusters.
4.  **Merging**: The system identifies split clusters that belong to the same entity and merges them.
5.  **Validation**: MDG checks ensure structural integrity throughout the process.
6.  **Reporting**: Final metrics and logs are output to the console and `logs/` directory.

---

## 📊 Output & Logging

### Console Output
You will see real-time progress bars, stage completion status, and a final summary:

```text
========================================
FINAL METRICS REPORT
========================================
Purity:         0.9850
Inverse Purity: 0.9920
F-Measure:      0.9885
ARI:            0.9750
----------------------------------------
Total API Calls:     145
Total Execution Time: 45.20 s
Total Tokens:        12500
Total MDG Interventions: 3
========================================
```

### Log Files
Every run generates a detailed log file in `logs/`, named with the dataset and timestamp:
`logs/demo_er_dataset_run_20260111_160914.log`

---

## 🛠️ Advanced Tuning

You can fine-tune the pipeline by modifying thresholds in `scripts/run_pipeline.py`:

*   `lsh_threshold`: Controls the strictness of the initial blocking.
*   `separation_threshold`: Controls when to split a cluster.
*   `block_threshold` & `merge_threshold`: Control the merging aggressiveness.



<p align="center">
  <i>Built with ❤️ by the LLMCER Team</i>
</p>
