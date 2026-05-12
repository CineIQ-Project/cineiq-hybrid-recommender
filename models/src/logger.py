"""Experiment logging with MLflow (with JSON file fallback)."""
import json
import os
import time
from datetime import datetime
from typing import Dict, Optional


class ExperimentLogger:
    """MLflow-style experiment logger with JSON fallback.

    Uses MLflow if available, otherwise logs to a JSON file.
    """

    def __init__(self, experiment_name: str = 'CineIQ_SVD',
                 tracking_uri: Optional[str] = None,
                 artifacts_dir: str = 'artifacts'):
        self._start_time = time.time()
        self._run_id: Optional[str] = None
        self._artifacts_dir = artifacts_dir
        self._log_data: Dict = {
            'experiment': experiment_name,
            'started_at': datetime.now().isoformat(),
            'params': {},
            'metrics': {},
            'artifacts': [],
            'dataset_info': {},
        }

        try:
            import mlflow
            self._mlflow = mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            run = mlflow.start_run()
            self._run_id = run.info.run_id
            print(f'  MLflow run started: {self._run_id}')
        except Exception as e:
            self._mlflow = None
            print(f'  MLflow unavailable ({e}), using JSON logging')

    def log_params(self, params: Dict) -> None:
        """Log hyperparameters."""
        self._log_data['params'].update(params)
        if self._mlflow:
            self._mlflow.log_params(params)
        print(f'  Logged {len(params)} params')

    def log_metrics(self, metrics: Dict) -> None:
        """Log evaluation metrics."""
        self._log_data['metrics'].update(metrics)
        if self._mlflow:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._mlflow.log_metric(k, v)
        print(f'  Logged {len(metrics)} metrics')

    def log_model(self, model_path: str) -> None:
        """Log model artifact path."""
        self._log_data['artifacts'].append(model_path)
        if self._mlflow:
            self._mlflow.log_artifact(model_path)
        print(f'  Logged model artifact')

    def log_dataset_info(self, info: Dict) -> None:
        """Log dataset statistics."""
        self._log_data['dataset_info'].update(info)
        if self._mlflow:
            for k, v in info.items():
                self._mlflow.log_param(f'data_{k}', v)

    def end_run(self) -> str:
        """End the logging run and save JSON log.

        Returns:
            Path to JSON log file
        """
        elapsed = time.time() - self._start_time
        self._log_data['duration_seconds'] = round(elapsed, 2)
        self._log_data['ended_at'] = datetime.now().isoformat()

        if self._mlflow:
            self._mlflow.end_run()
            print(f'  MLflow run ended: {self._run_id}')

        # Always save JSON log as backup
        log_path = os.path.join(self._artifacts_dir, 'experiment_log.json')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(self._log_data, f, indent=2, default=str)
        print(f'  JSON log saved: {log_path}')
        return log_path
