from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator # (Was DummyOperator in older versions)
from airflow.models.param import Param
from datetime import datetime
import os

# --- Configurations ---
PROJECT_ROOT = "/opt/airflow" 
QUALITY_THRESHOLD = 0.15 # Minimum BLEU score required to merge

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

# --- Python Function for the Branching Logic ---
def check_model_quality(**kwargs):
    """
    Reads the eval_results.txt file created by the evaluate task.
    Decides whether to merge or stop based on the score.
    """
    # 1. Get the adapter path from the DAG params
    adapter_name = kwargs['dag_run'].conf.get('adapter_name')
    results_path = f"{PROJECT_ROOT}/models/adapters/{adapter_name}/eval_results.txt"
    
    print(f"Checking results at: {results_path}")
    
    try:
        # 2. Read the file (Format: "BLEU: 0.25\nBERTScore: 0.88")
        scores = {}
        with open(results_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if ':' in line:
                    key, val = line.strip().split(':')
                    scores[key.strip()] = float(val.strip())
            
        bleu_score = scores.get('BLEU', 0.0)
        bert_score = scores.get('BERTScore', 0.0)
            
        print(f"Detected Scores - BLEU: {bleu_score}, BERTScore: {bert_score}")
        print(f"Thresholds - BLEU: {QUALITY_THRESHOLD}")

        # 3. The Decision (Currently just checks BLEU, but ready for BERTScore logic)
        if bleu_score >= QUALITY_THRESHOLD:
            print("✅ Quality Check PASSED. Proceeding to Merge.")
            return 'merge_adapter'
        else:
            print("❌ Quality Check FAILED. Stopping pipeline.")
            return 'stop_low_quality'
            
    except Exception as e:
        print(f"Error reading results: {e}")
        # If we can't read the score, we should probably fail safe and stop
        return 'stop_low_quality'

# --- The DAG Definition ---
with DAG(
    'continuous_finetuning_pipeline',
    default_args=default_args,
    schedule_interval=None, 
    catchup=False,
    params={
        "data_path": Param(default="data/processed/batch_2_logistics.jsonl", type="string", description="Training Data Path"),
        "adapter_name": Param(default="adapter_v2_docker", type="string", description="Adapter Output Name"),
    },
) as dag:

    # 1. Start
    start = BashOperator(
        task_id='start_pipeline',
        bash_command='echo "Starting MLOps Pipeline with Quality Gate..."',
    )

    # 2. Train
    train_model = BashOperator(
        task_id='train_adapter',
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        python src/scripts/train.py \
        {{{{ dag_run.conf.get('data_path') }}}} \
        {{{{ dag_run.conf.get('adapter_name') }}}}
        """,
    )

    # 3. Evaluate
    # This script writes 'eval_results.txt' which the next task reads
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        python src/scripts/evaluates.py \
        {{{{ dag_run.conf.get('adapter_name') }}}}
        """,
        # Note: evaluates.py only takes adapter_name now and calculates paths internally
    )

    # 4. Quality Gate (The New Logic)
    quality_check = BranchPythonOperator(
        task_id='quality_check',
        python_callable=check_model_quality,
        provide_context=True,
    )

    # 5a. Branch A: Merge (Success)
    merge_adapter = BashOperator(
        task_id='merge_adapter',
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        python src/scripts/merge.py \
        {PROJECT_ROOT}/models/adapters/{{{{ dag_run.conf.get('adapter_name') }}}} \
        {PROJECT_ROOT}/models/adapters/merged_{{{{ dag_run.conf.get('adapter_name') }}}}
        """,
    )

    # 5b. Branch B: Stop (Failure)
    stop_low_quality = EmptyOperator(
        task_id='stop_low_quality'
    )

    # 6. Deploy (Only happens after Merge)
    deploy_model = BashOperator(
        task_id='deploy_to_vllm',
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        python src/scripts/deploy.py \
        {{{{ dag_run.conf.get('adapter_name') }}}} \
        {PROJECT_ROOT}/models/adapters/merged_{{{{ dag_run.conf.get('adapter_name') }}}}
        """,
        trigger_rule='none_failed' # Only run if upstream didn't fail
    )


    # --- Define the Flow ---
    # 1. Linear part
    start >> train_model >> evaluate_model >> quality_check
    
    # 2. Branching part
    quality_check >> merge_adapter >> deploy_model  # Path A (Success)
    quality_check >> stop_low_quality               # Path B (Failure)