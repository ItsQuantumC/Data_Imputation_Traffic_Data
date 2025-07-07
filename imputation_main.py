# main.py

import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from data_loader import load_and_prepare_data
from mrnn_bd_gru import MRNN as AdvancedMRNN
from mrnn_baseline import MRNN as BaselineMRNN
from utils import calculate_metrics, denormalize
from custom_logger import UnifiedJSONLogger, EpochDataCallback, NpEncoder


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
        unified_logger.start_run(run+1, run_seed)
        print(f"\n--- Starting Run {run + 1}/{args.num_runs} (Seed: {run_seed}) ---")

        data = load_and_prepare_data(
            file_path=args.file_name,
            seq_len=args.seq_len,
            artificial_missing_rate=args.missing_rate,
            grin_path=args.grin_path, 
            mean_path=args.mean_path
        )
        
        train_x, train_m, train_t, train_ori_norm = data["train"]
        valid_x, valid_m, valid_t, valid_ori_norm = data["valid"]
        test_x, test_m, test_t, test_ori_norm = data["test"]
        artificial_test_mask = data["test_masks"]["artificial"]
        norm_params = data["norm_params"]
        train_norm_means = data["train_norm_means"]

        if test_x.shape[0] == 0:
            print("Not enough test data to create sequences. Skipping run.")
            continue

        data_module = pl.LightningDataModule()
        data_module.setup = lambda stage: None
        
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_x).float(), torch.from_numpy(train_m).float(),
            torch.from_numpy(train_t).float(), torch.from_numpy(train_ori_norm).float()
        )
        valid_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(valid_x).float(), torch.from_numpy(valid_m).float(),
            torch.from_numpy(valid_t).float(), torch.from_numpy(valid_ori_norm).float()
        )
        
        data_module.train_dataloader = lambda: torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        data_module.val_dataloader = lambda: torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        
        if args.use_baseline:
            model = BaselineMRNN(dim=train_x.shape[2], h_dim=args.h_dim, learning_rate=args.learning_rate)  # Model #1: Baseline MRNN
            model_name = "baseline_MRNN"
        else:
            model = AdvancedMRNN(dim=train_x.shape[2], h_dim=args.h_dim, learning_rate=args.learning_rate)    # Model #2: GRU-based Imputation
            model_name = "advanced_MRNN"
        
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('medium')
        
        epoch_logger_callback = EpochDataCallback(unified_logger)

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            enable_model_summary=False,
            enable_checkpointing=False,
            callbacks=[epoch_logger_callback, pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
            logger=False
        )
        start_time = time.time()
        trainer.fit(model, data_module)
        training_time = time.time() - start_time

        start_pred_time = time.time()
        imputed_x_norm = model.impute(test_x, test_m, test_t)
        prediction_time = time.time() - start_pred_time

        means_broadcastable = np.tile(train_norm_means, (test_ori_norm.shape[0], test_ori_norm.shape[1], 1))
        mean_imputed_norm = np.where(artificial_test_mask.astype(bool), means_broadcastable, test_ori_norm)
        imputed_x, ground_truth_x, mean_imputed_x = map(
            lambda x_data: denormalize(x_data, norm_params),
            [imputed_x_norm, test_ori_norm, mean_imputed_norm]
        )

        imputed_x_flat = imputed_x.reshape(-1, imputed_x.shape[2])
        imputed_df = pd.DataFrame(imputed_x_flat)
        # imputed_df.to_csv(args.output_file.replace('.json', f'_imputed_run_{run+1}.csv'), index=False)
        imputed_df.to_csv(args.output_file.replace('.json', f'_{model_name}_run_{run+1}.csv'), index=False)

        mrnn_scores = calculate_metrics(ground_truth_x, imputed_x, artificial_test_mask)
        grin_scores = {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}
        mean_scores = calculate_metrics(ground_truth_x, mean_imputed_x, artificial_test_mask)
        
        run_scores = {"MRNN": mrnn_scores, "GRIN": grin_scores, "Mean": mean_scores}
        run_runtimes = {
            "total_train_time": training_time,
            "avg_epoch_time": training_time / (trainer.current_epoch + 1) if trainer.current_epoch >= 0 else 0,
            "prediction_time": prediction_time
        }

        unified_logger.end_run(scores=run_scores, runtimes=run_runtimes)
        
        print(f"Run {run + 1} MRNN Scores: {mrnn_scores}")
        print(f"Run {run + 1} Mean Scores: {mean_scores}")

    unified_logger.summarize_results(models_to_summarize=["MRNN", "GRIN", "Mean"])
    unified_logger.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="data/timeseries_data_with-NaN.csv", type=str)
    parser.add_argument("--grin_path", default="data/imputed_dataset.csv", type=str)
    parser.add_argument("--mean_path", default="data/mean_imputed.csv", type=str)
    parser.add_argument("--output_file", default="data/outputs/imputation_evaluation_results.json", type=str)
    parser.add_argument("--seq_len", default=20, type=int)
    parser.add_argument("--missing_rate", default=0.3, type=float)
    parser.add_argument("--h_dim", default=256, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--num_runs", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--use_baseline", default=False, action='store_true',
                        help="Use the baseline MRNN model instead of the advanced GRU-based model.")
    args = parser.parse_args()
    
    main(args)
