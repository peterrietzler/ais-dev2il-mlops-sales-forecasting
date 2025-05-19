from datetime import datetime, timedelta

from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.job import (
    KubernetesJobOperator
)

def _copy_file_and_replace_store_id(input_file, output_file, store_id):
    with open(input_file, 'r') as file:
        data = file.read()
    data = data.replace("__STORE_ID__", f"{store_id}")
    with open(output_file, 'w') as file:
        file.write(data)


with DAG(
    "sales-forecasting-training",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function, # or list of functions
        # 'on_success_callback': some_other_function, # or list of functions
        # 'on_retry_callback': another_function, # or list of functions
        # 'sla_miss_callback': yet_another_function, # or list of functions
        # 'on_skipped_callback': another_function, #or list of functions
        # 'trigger_rule': 'all_success'
    },
    #schedule=timedelta(days=1),
    #start_date=datetime(2021, 1, 1),
    #catchup=False
) as dag:

    download_latest_sales_data = KubernetesJobOperator(
        task_id="download-latest-sales-data",
        job_template_file="/opt/airflow/dags/k8s-jobs/download-latest-sales-data-job.yaml",
        wait_until_job_complete=True,
        retries=0
    )
    
    for store_id in [1, 2, 3, 4]:
        _copy_file_and_replace_store_id("/opt/airflow/dags/k8s-jobs/train-store-model-job.yaml",
                                        f"/opt/airflow/dags/k8s-jobs/train-store-{store_id}-model-job.yaml", 
                                        store_id)
        train_store_model = KubernetesJobOperator(
            task_id=f"train-store-model-{store_id}",
            job_template_file=f"/opt/airflow/dags/k8s-jobs/train-store-{store_id}-model-job.yaml",
            wait_until_job_complete=True,
            retries=0,
            env_vars={"STORE_ID": str(store_id)}
        )
        download_latest_sales_data >> train_store_model