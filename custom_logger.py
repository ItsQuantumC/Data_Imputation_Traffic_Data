# seminar/custom_logger.py

import json
import time
import numpy as np
import pytorch_lightning as pl

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class EpochDataCallback(pl.Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        train_loss = trainer.callback_metrics.get('train_loss_epoch', None)
        if train_loss is not None:
            self.logger.log_epoch_metric(epoch, 'train_loss', train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = pl_module.current_epoch
        val_loss = trainer.callback_metrics.get('val_loss', None)
        if val_loss is not None:
            self.logger.log_epoch_metric(epoch, 'val_loss', val_loss.item())


class UnifiedJSONLogger:
    def __init__(self, output_file, hyperparameters, hardware_info):
        self.output_file = output_file
        self.results = {
            "hyperparameters": hyperparameters,
            "hardware": hardware_info,
            "runs": [],
            "summary": {}
        }
        self.current_run_data = None
        self.current_epoch_logs = []

    def start_run(self, run_id, seed):
        """Initializes a new run."""
        self.current_run_data = {
            "run": run_id,
            "seed": seed,
            "epoch_logs": [],
            "scores": {},
            "runtimes": {}
        }
        self.current_epoch_logs = []
        print(f"\n--- Starting Run {run_id} (Seed: {seed}) ---")

    def log_epoch_metric(self, epoch, name, value):
        epoch_log = next((log for log in self.current_epoch_logs if log['epoch'] == epoch), None)
        if epoch_log:
            epoch_log[name] = value
        else:
            self.current_epoch_logs.append({'epoch': epoch, name: value})

    def end_run(self, scores, runtimes):
        if self.current_run_data is None:
            raise Exception("Cannot end run before starting one.")
        self.current_run_data["epoch_logs"] = sorted(self.current_epoch_logs, key=lambda x: x['epoch'])
        self.current_run_data["scores"] = scores
        self.current_run_data["runtimes"] = runtimes
        self.results["runs"].append(self.current_run_data)
        self.current_run_data = None
        self.current_epoch_logs = []

    def summarize_results(self, models_to_summarize=["MRNN", "GRIN", "Mean"]):
        if not self.results["runs"]:
            return

        for model_name in models_to_summarize:
            self.results["summary"][model_name] = {}
            if model_name in self.results["runs"][0]["scores"]:
                for metric in self.results["runs"][0]["scores"][model_name].keys():
                    scores = [r["scores"][model_name][metric] for r in self.results["runs"]]
                    self.results["summary"][model_name][f"{metric}_mean"] = np.mean(scores)
                    self.results["summary"][model_name][f"{metric}_std"] = np.std(scores)

    def summarize_forecasting_results(self):
        if not self.results["runs"]:
            return
        if "scores" in self.results["runs"][0]:
            for metric in self.results["runs"][0]["scores"].keys():
                scores = [r["scores"][metric] for r in self.results["runs"]]
                self.results["summary"][f"{metric}_mean"] = np.mean(scores)
                self.results["summary"][f"{metric}_std"] = np.std(scores)


    def save(self):
        print("\n--- Final Evaluation Summary ---")
        print(json.dumps(self.results["summary"], indent=4))
        with open(self.output_file, 'w') as f:
            json.dump(self.results, f, indent=4, cls=NpEncoder)
        print(f"\n✅ All results, including per-epoch logs, saved to {self.output_file}")
