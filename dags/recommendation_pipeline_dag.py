"""Apache Airflow (2.x) version of the SAME workflow defined in src/orchestration/pipeline.py.

Tasks and dependencies are generated from `build_dag()`, so the Airflow DAG and the local
runner can never diverge.  Deploy by pointing AIRFLOW__CORE__DAGS_FOLDER at this `dags/` folder
(or symlinking this file) with the repo root on PYTHONPATH.

    airflow dags test recommendation_pipeline 2025-01-01      # run once from the CLI

NOTE: Airflow could not be installed in the offline environment where this repo was built, so this
file was syntax-checked and its dependency graph is verified in tests/test_pipeline.py, but it was NOT
executed inside an Airflow scheduler.  The identical tasks were executed by the local runner (run_pipeline.py).
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airflow import DAG                                   # noqa: E402
from airflow.operators.python import PythonOperator       # noqa: E402

from src.orchestration.pipeline import build_dag          # noqa: E402
from src.utils import load_config                         # noqa: E402

default_args = {"owner": "data-team", "retries": 1, "retry_delay": timedelta(minutes=2)}

with DAG(
    dag_id="recommendation_pipeline",
    description="Ingest -> validate -> prepare -> split -> features -> train -> evaluate",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["recsys", "dm4ml"],
) as dag:
    operators = {}
    for task in build_dag():
        operators[task.name] = PythonOperator(
            task_id=task.name,
            python_callable=lambda fn=task.fn: fn(load_config()),
        )
    for task in build_dag():
        for dep in task.deps:
            operators[dep] >> operators[task.name]
