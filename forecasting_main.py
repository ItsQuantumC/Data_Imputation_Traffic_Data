# forecasting_main.py

import argparse
import json
import time
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader

from forecasting_data_loader import load_and_prepare_forecasting_data
from forecasting_model import Seq2Seq
from utils import calculate_metrics, denormalize
from custom_logger import UnifiedJSONLogger, EpochDataCallback, NpEncoder

log_dir = "forecasting_logs"

def main(args):
    hardware_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }
    print(f"Hardware Info: {json.dumps(hardware_info, indent=4)}")

    unified_logger = UnifiedJSONLogger(
        output_file=args.output_file,
        hyperparameters=vars(args),
        hardware_info=hardware_info
    )

    for run in range(args.num_runs):
        run_seed = args.seed + run
        pl.seed_everything(run_seed)
        unified_logger.start_run(run + 1, run_seed)
        print(f"\n--- Starting Run {run + 1}/{args.num_runs} (Seed: {run_seed}) ---")

        data = load_and_prepare_forecasting_data(
            imputed_file_path=args.imputed_file,
            original_data_file_path=args.original_data_file,
            input_len=args.input_len,
            output_len=args.output_len
        )
        
        train_x, train_y = data["train"]
        valid_x, valid_y = data["valid"]
        test_x, test_y = data["test"]
        norm_params = data["norm_params"]

        train_dataset = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
        valid_dataset = TensorDataset(torch.from_numpy(valid_x).float(), torch.from_numpy(valid_y).float())
        test_dataset = TensorDataset(torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float())

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        model = Seq2Seq(
            input_dim=train_x.shape[2],
            hidden_dim=args.h_dim,
            output_dim=train_y.shape[2],
            n_layers=args.n_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate
        )

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('medium')
        
        epoch_logger_callback = EpochDataCallback(unified_logger)

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[epoch_logger_callback, pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False
        )
        
        start_time = time.time()
        trainer.fit(model, train_loader, valid_loader)
        training_time = time.time() - start_time
        
        start_pred_time = time.time()
        prediction_results = trainer.predict(model, test_loader)

        predictions_norm = np.concatenate([r['predictions'] for r in prediction_results])
        ground_truth_norm = np.concatenate([r['ground_truth'] for r in prediction_results])
        prediction_time = time.time() - start_pred_time
        
        predictions = denormalize(predictions_norm, norm_params)
        ground_truth = denormalize(ground_truth_norm, norm_params)

        eval_mask = np.ones_like(ground_truth, dtype=bool)
        
        run_scores = calculate_metrics(ground_truth, predictions, eval_mask)
        
        run_runtimes = {
            "total_train_time": training_time,
            "avg_epoch_time": training_time / (trainer.current_epoch + 1) if trainer.current_epoch >= 0 else 0,
            "prediction_time": prediction_time
        }

        unified_logger.end_run(scores=run_scores, runtimes=run_runtimes)
        print(f"Run {run + 1} Scores: {run_scores}")

    unified_logger.summarize_forecasting_results()
    unified_logger.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data_file", default="data/timeseries_data_with-NaN.csv", type=str, help="Path to original data file with timestamps.")
    parser.add_argument("--imputed_file", default="data/imputed_dataset.csv", type=str)
    parser.add_argument("--output_file", default="data/forecasting_evaluation_results.json", type=str)
    parser.add_argument("--input_len", default=20, type=int)
    parser.add_argument("--output_len", default=1, type=int)
    parser.add_argument("--h_dim", default=256, type=int)
    parser.add_argument("--n_layers", default=8, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--num_runs", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    
    main(args)
