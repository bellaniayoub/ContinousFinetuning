from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, ShortCircuitOperator
from airflow.operators.empty import EmptyOperator
from airflow.models.param import Param
from datetime import datetime
import os
import glob
import shutil

# --- Configurations ---
PROJECT_ROOT = "/opt/airflow" 
QUALITY_THRESHOLD = 0.15 

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 0,
}

# --- 1. Sensing Logic ---
def sense_and_move_file(**kwargs):
    """
    Checks data/input for new JSONL files.
    If found: Moves to data/processed and pushes paths to XCom.
    If empty: Returns False (skips pipeline).
    """
    input_dir = f"{PROJECT_ROOT}/data/input"
    files = glob.glob(f"{input_dir}/*.jsonl")
    
    if not files:
        print("No new files found. Skipping.")
        return False
    
    # Take the first file
    file_path = files[0]
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]
    
    # Generate timestamped name to avoid collisions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adapter_name = f"adapter_{base_name}_{timestamp}"
    new_filename = f"{base_name}_{timestamp}.jsonl"
    
    processed_dir = f"{PROJECT_ROOT}/data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    new_path = f"{processed_dir}/{new_filename}"
    
    print(f"Moving {file_path} -> {new_path}")
    shutil.move(file_path, new_path)
    
    # Push to XCom
    ti = kwargs['ti']
    ti.xcom_push(key='data_path', value=f"data/processed/{new_filename}")
    ti.xcom_push(key='adapter_name', value=adapter_name)
    
    return True

# --- 2. Quality Check Logic ---
def check_model_quality(**kwargs):
    ti = kwargs['ti']
    # Try getting from XCom first (Auto mode), else Param (Manual mode)
    adapter_name = ti.xcom_pull(key='adapter_name', task_ids='sense_file')
    if not adapter_name:
        adapter_name = kwargs['dag_run'].conf.get('adapter_name')

    results_path = f"{PROJECT_ROOT}/models/adapters/{adapter_name}/eval_results.txt"
    
    print(f"Checking results at: {results_path}")
    
    try:
        scores = {}
        with open(results_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if ':' in line:
                    key, val = line.strip().split(':')
                    scores[key.strip()] = float(val.strip())
            
        bleu_score = scores.get('BLEU', 0.0)
        print(f"Detected BLEU: {bleu_score} (Threshold: {QUALITY_THRESHOLD})")

        if bleu_score >= QUALITY_THRESHOLD:
            return 'merge_adapter'
        else:
            return 'stop_low_quality'
            
    except Exception as e:
        print(f"Error reading results: {e}")
        return 'stop_low_quality'

# --- The DAG Definition ---
with DAG(
    'continuous_finetuning_pipeline',
    default_args=default_args,
    schedule_interval='*/5 * * * *', # Run every 5 minutes
    catchup=False,
    params={
        "data_path": Param(default="data/processed/default.jsonl", type="string", description="Training Data Path (Manual Trigger Only)"),
        "adapter_name": Param(default="manual_adapter", type="string", description="Adapter Output Name (Manual Trigger Only)"),
    },
) as dag:

    # 1. Sense File (ShortCircuit)
    sense_file = ShortCircuitOperator(
        task_id='sense_file',
        python_callable=sense_and_move_file,
        provide_context=True,
    )

    # 2. Train (Dynamic Params)
    train_model = BashOperator(
        task_id='train_adapter',
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        DATA_PATH="{{{{ ti.xcom_pull(key='data_path', task_ids='sense_file') or dag_run.conf.get('data_path') }}}}" && \
        ADAPTER_NAME="{{{{ ti.xcom_pull(key='adapter_name', task_ids='sense_file') or dag_run.conf.get('adapter_name') }}}}" && \
        echo "Training on: $DATA_PATH as $ADAPTER_NAME" && \
        python src/scripts/train.py $DATA_PATH $ADAPTER_NAME
        """,
    )

    # 3. Evaluate
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        ADAPTER_NAME="{{{{ ti.xcom_pull(key='adapter_name', task_ids='sense_file') or dag_run.conf.get('adapter_name') }}}}" && \
        python src/scripts/evaluates.py $ADAPTER_NAME
        """,
    )

    # 4. Quality Gate
    quality_check = BranchPythonOperator(
        task_id='quality_check',
        python_callable=check_model_quality,
        provide_context=True,
    )

    # 5a. Merge
    merge_adapter = BashOperator(
        task_id='merge_adapter',
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        ADAPTER_NAME="{{{{ ti.xcom_pull(key='adapter_name', task_ids='sense_file') or dag_run.conf.get('adapter_name') }}}}" && \
        python src/scripts/merge.py \
        {PROJECT_ROOT}/models/adapters/$ADAPTER_NAME \
        {PROJECT_ROOT}/models/adapters/merged_$ADAPTER_NAME
        """,
    )

    # 5b. Stop
    stop_low_quality = EmptyOperator(task_id='stop_low_quality')

    # 6. Deploy
    deploy_model = BashOperator(
        task_id='deploy_to_vllm',
        bash_command=f"""
        cd {PROJECT_ROOT} && \
        ADAPTER_NAME="{{{{ ti.xcom_pull(key='adapter_name', task_ids='sense_file') or dag_run.conf.get('adapter_name') }}}}" && \
        python src/scripts/deploy.py \
        $ADAPTER_NAME \
        {PROJECT_ROOT}/models/adapters/merged_$ADAPTER_NAME
        """,
        trigger_rule='none_failed'
    )

    # --- Flow ---
    sense_file >> train_model >> evaluate_model >> quality_check
    quality_check >> merge_adapter >> deploy_model
    quality_check >> stop_low_quality