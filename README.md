# End-to-End MLOps Pipeline for LLM Fine-Tuning

## 1. Project Overview
This project establishes a robust **Continuous Fine-Tuning Pipeline** for Large Language Models (LLMs). It automates the lifecycle of adapting an open-source model (`Qwen2.5-1.5B-Instruct`) to specific customer service tasks using **QLoRA** (Quantized Low-Rank Adaptation).

The system uses **Apache Airflow** for orchestration, **vLLM** for high-performance inference, and **Docker** for containerization, ensuring a reproducible and scalable environment.

## 2. System Architecture

The project consists of four main Docker services orchestrated via `docker-compose`:

1.  **Airflow (Scheduler & Webserver)**: Manages the ML pipeline DAG (Directed Acyclic Graph).
2.  **vLLM Server**: An OpenAI-compatible, high-throughput model serving engine.
3.  **Chat UI (Streamlit)**: A user-friendly web interface for interacting with the model.
4.  **FastAPI Backend**: A REST API layer for programmatic access.
5.  **PostgreSQL**: The database backend for Airflow.

### Data Flow
1.  **Ingestion**: Raw JSONL data is processed.
2.  **Training**: Airflow triggers `train.py` to fine-tune the base model.
3.  **Evaluation**: `evaluates.py` computes metrics (BLEU, BERTScore).
4.  **Quality Gate**: If metrics meet the threshold, the pipeline proceeds.
5.  **Merging**: The LoRA adapter is fused with the base model.
6.  **Deployment**:
    *   **Remote**: The merged model is uploaded to Hugging Face Hub.
    *   **Local**: A symbolic link (`production`) is updated to point to the new model.
7.  **Inference**: vLLM detects the updated symlink (after restart) and serves the new model to the UI/API.

## 3. Prerequisites

*   **OS**: Windows 10/11 with **WSL2** (Ubuntu) or Linux.
*   **GPU**: NVIDIA GPU with drivers installed (CUDA support).
*   **Docker Desktop**: Configured to use the WSL2 backend.
*   **Hugging Face Account**: API Token with **Write** permissions.

## 4. Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository_url>
cd MLopsProject
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
HF_TOKEN=your_hugging_face_write_token_here
```

### 3. Build the Infrastructure
Run the following command to build and start all containers:
```bash
docker-compose up -d --build
```
*This may take several minutes as it downloads the base model and builds the Docker images.*

## 5. The MLOps Pipeline Details

The Airflow DAG (`continuous_finetuning_pipeline`) executes the following steps:

### Step 1: Training (`train.py`)
Fine-tunes the model using the `SFTTrainer` from Hugging Face `trl`.
- **Technique**: QLoRA (4-bit quantization).
- **Output**: LoRA Adapter saved to `models/adapters/{adapter_name}`.

### Step 2: Evaluation (`evaluates.py`)
Generates responses on a test set and calculates quality metrics.
- **Metrics**: BLEU Score, BERTScore.
- **Output**: `eval_results.txt`.

### Step 3: Quality Gate (`check_model_quality`)
A branching task that reads `eval_results.txt`.
- **Pass**: If BLEU > Threshold (0.15), proceed to Merge.
- **Fail**: Stop the pipeline to prevent deploying bad models.

### Step 4: Merge (`merge.py`)
Fuses the fine-tuned adapter with the base model to create a standalone model.
- **Why**: Faster inference (no adapter overhead) and easier compatibility.
- **Output**: `models/adapters/merged_{adapter_name}`.

### Step 5: Deployment (`deploy.py`)
- **Action**: Uploads the merged folder to your Hugging Face Hub repository.
- **Action**: Updates the local symbolic link `models/adapters/production` to point to the new merge.

*(Screens: Include screenshots of the Airflow Graph View and Grid View here)*

## 6. Interfaces

### Streamlit Chat UI
A visual chat interface to test the model conversationally.
- **URL**: [http://localhost:8501](http://localhost:8501)
- **Features**: Chat history, system prompt configuration, temperature slider.

*(Screens: Include a screenshot of the Streamlit UI chatting with the bot)*

### FastAPI Backend
A RESTful API endpoint for integrations.
- **Documentation**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **Endpoint**: `POST /chat`

*(Screens: Include a screenshot of the Swagger UI or a curl request)*

## 7. How to Run the Project (Tutorial)

1.  **Start the Services**:
    Ensure Docker is running and execute:
    ```bash
    docker-compose up -d
    ```

2.  **Access Airflow**:
    - Go to [http://localhost:8080](http://localhost:8080).
    - Login with `admin` / `admin`.
    - Enable the `continuous_finetuning_pipeline` DAG.
    - Click the **Trigger DAG** button (Release Play button).

3.  **Monitor Progress**:
    - Watch the tasks turn dark green (Success) in the Airflow Grid View.
    - Check the logs of the `train_adapter` task to see training progress.

4.  **Verify Deployment**:
    - Once the `deploy_to_vllm` task succeeds, check your Hugging Face account for the new model repo.
    - Provide a restart to vLLM to pick up the new model (if not handled automatically): `docker-compose restart vllm_server`.

5.  **Test the Model**:
    - Open the Streamlit UI [http://localhost:8501](http://localhost:8501) and ask a question relevant to your dataset.

## 8. Artifacts & file Structure
```
MLopsProject/
├── airflow/
│   └── dags/
│       └── countinous_finetuning.py  # Airflow Pipeline Definition
├── data/
│   └── processed/                    # Training Data
├── models/
│   └── adapters/                     # Saved Models & Symlinks
├── src/
│   ├── api/
│   │   └── main.py                   # FastAPI Backend
│   ├── scripts/
│   │   ├── train.py                  # Training Script
│   │   ├── evaluates.py              # Evaluation Script
│   │   ├── merge.py                  # Merge Script
│   │   └── deploy.py                 # Deployment Script
│   └── ui/
│       └── app.py                    # Streamlit UI
├── docker-compose.yml                # Service Orchestration
├── Dockerfile                        # Airflow Image Definition
└── README.md                         # This Report
```
