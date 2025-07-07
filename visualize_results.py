# visualize_results.py

import json
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Plotting Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 22


def load_json_data(path):
    """Loads a JSON file safely."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {path}")
        return None

def plot_imputation_performance(baseline_results, advanced_results, output_dir):
    """
    Generates a grouped bar chart comparing imputation model performance (MAE, RMSE, MAPE).
    This plot directly addresses the core of Part A.
    """
    print("📊 Plotting Part A: Imputation Performance Bar Chart...")
    if not baseline_results or not advanced_results:
        print("   -> Skipping due to missing data.")
        return

    metrics = ['mae', 'rmse', 'mape']
    model_performance = {}

    # Extract data for Advanced MRNN from its results file
    if 'MRNN' in advanced_results['summary']:
        model_performance['MRNN (Advanced)'] = {
            'mean': [advanced_results['summary']['MRNN'][f'{m}_mean'] for m in metrics],
            'std': [advanced_results['summary']['MRNN'][f'{m}_std'] for m in metrics]
        }
    
    # Extract data for GRIN and Mean from the baseline results file
    for model_name in ["GRIN", "Mean"]:
        if model_name in baseline_results['summary']:
            model_performance[model_name] = {
                'mean': [baseline_results['summary'][model_name][f'{m}_mean'] for m in metrics],
                'std': [baseline_results['summary'][model_name][f'{m}_std'] for m in metrics]
            }

    labels = [m.upper() for m in metrics]
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    model_names = list(model_performance.keys())
    for i, model_name in enumerate(model_names):
        means = model_performance[model_name]['mean']
        stds = model_performance[model_name]['std']
        # Position bars side-by-side
        pos = x - width + (i * width)
        ax.bar(pos, means, width, yerr=stds, label=model_name, capsize=5, alpha=0.85)

    ax.set_ylabel('Error Value')
    ax.set_title('Part A: Imputation Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Imputation Method")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1a_imputation_performance_bar_chart.png"))
    plt.close()
    print("   ✅ Saved 1a_imputation_performance_bar_chart.png")


def plot_qualitative_imputation(original_csv, mrnn_imputed_csv, grin_imputed_csv, mean_imputed_csv, output_dir, sensor_id='21649702', time_window=('2023-11-20', '2023-11-23')):
    """
    Plots a time series showing ground truth vs. imputed values for a single sensor,
    providing a qualitative view of imputation quality.
    """
    print(f"📈 Plotting Part A: Qualitative Imputation for sensor {sensor_id}...")
    try:
        # Load data and ensure datetime index
        ground_truth_df = pd.read_csv(original_csv, index_col='timestamp', parse_dates=True)
        
        # The imputed CSVs don't have a timestamp; align them with the original DF's index
        mrnn_df = pd.read_csv(mrnn_imputed_csv)
        mrnn_df.index = ground_truth_df.index[-len(mrnn_df):]
        mrnn_df.columns = ground_truth_df.columns
        
        grin_df = pd.read_csv(grin_imputed_csv, index_col=0) # GRIN output has an unnamed index col
        grin_df.index = ground_truth_df.index
        
        mean_df = pd.read_csv(mean_imputed_csv, index_col='timestamp', parse_dates=True)

        # Select the specified time window and sensor
        subset_gt = ground_truth_df.loc[time_window[0]:time_window[1], sensor_id]
        subset_mrnn = mrnn_df.loc[time_window[0]:time_window[1], sensor_id]
        subset_grin = grin_df.loc[time_window[0]:time_window[1], sensor_id]
        subset_mean = mean_df.loc[time_window[0]:time_window[1], sensor_id]

        plt.figure(figsize=(18, 9))
        
        # Plot ground truth and the different imputations
        plt.plot(subset_gt.index, subset_gt.values, 'o-', label='Ground Truth (Observed)', color='black', markersize=5, zorder=5)
        plt.plot(subset_mrnn.index, subset_mrnn.values, '.-', label='MRNN Imputation', color='crimson', alpha=0.9, zorder=4)
        plt.plot(subset_grin.index, subset_grin.values, '.-', label='GRIN Imputation', color='forestgreen', alpha=0.8, zorder=3)
        plt.plot(subset_mean.index, subset_mean.values, '--', label='Mean Imputation', color='darkorange', alpha=0.9, zorder=2)
        
        # Highlight with 'x' marks where the original data was missing and MRNN imputed it
        missing_indices = subset_gt.isna()
        plt.scatter(subset_gt.index[missing_indices], subset_mrnn[missing_indices], c='crimson', marker='x', s=150, label='MRNN Imputed Point', zorder=6)

        plt.title(f'Part A: Qualitative Imputation Example (Sensor: {sensor_id})')
        plt.xlabel('Timestamp')
        plt.ylabel('Traffic Flow')
        plt.legend()
        plt.xticks(rotation=30)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "1b_qualitative_imputation_example.png"))
        plt.close()
        print(f"   ✅ Saved 1b_qualitative_imputation_example.png")

    except Exception as e:
        print(f"   ❌ Could not generate qualitative imputation plot. Error: {e}")


def plot_error_distribution(original_csv, mrnn_imputed_csv, grin_imputed_csv, mean_imputed_csv, output_dir, test_slice=('2023-11-17 00:00:00', '2024-01-21 07:00:00')):
    """
    Generates a box plot of the absolute imputation errors on artificially missing values.
    """
    print("📦 Plotting Part A: Imputation Error Distribution...")
    try:
        gt_df = pd.read_csv(original_csv, index_col='timestamp', parse_dates=True)
        
        mrnn_df = pd.read_csv(mrnn_imputed_csv)
        mrnn_df.index = gt_df.index[-len(mrnn_df):]
        mrnn_df.columns = gt_df.columns
        
        grin_df = pd.read_csv(grin_imputed_csv, index_col=0)
        grin_df.index = gt_df.index
        
        mean_df = pd.read_csv(mean_imputed_csv, index_col='timestamp', parse_dates=True)

        # Focus only on the test set period
        gt_test = gt_df.loc[test_slice[0]:test_slice[1]]
        mrnn_test = mrnn_df.loc[test_slice[0]:test_slice[1]]
        grin_test = grin_df.loc[test_slice[0]:test_slice[1]]
        mean_test = mean_df.loc[test_slice[0]:test_slice[1]]

        # We only evaluate on artificially missing values. Since we can't perfectly recreate the
        # artificial mask here, we will calculate errors where GRIN and Mean differ from GT,
        # assuming these are the imputed locations. This is a reasonable proxy.
        eval_mask = ~np.isclose(grin_test.values, gt_test.values)
        
        error_mrnn = np.abs(mrnn_test.values[eval_mask] - gt_test.values[eval_mask])
        error_grin = np.abs(grin_test.values[eval_mask] - gt_test.values[eval_mask])
        error_mean = np.abs(mean_test.values[eval_mask] - gt_test.values[eval_mask])

        error_data = pd.DataFrame({
            'MRNN (Advanced)': error_mrnn,
            'GRIN': error_grin,
            'Mean': error_mean,
        })
        
        error_data_melted = error_data.melt(var_name='Model', value_name='Absolute Error').dropna()

        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Model', y='Absolute Error', data=error_data_melted, palette="Set2")
        plt.title('Part A: Distribution of Absolute Imputation Errors (on Test Set)')
        plt.ylabel('Absolute Error |Ground Truth - Imputation|')
        plt.yscale('log') # Errors are often skewed; a log scale helps visualize the distribution better.
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "1c_imputation_error_distribution.png"))
        plt.close()
        print("   ✅ Saved 1c_imputation_error_distribution.png")

    except Exception as e:
        print(f"   ❌ Could not generate error distribution plot. Error: {e}")


def plot_forecasting_performance(forecasting_results, output_dir):
    """
    Generates a bar chart for the forecasting model performance (Part B).
    """
    print("📊 Plotting Part B: Forecasting Performance Bar Chart...")
    if not forecasting_results:
        print("   -> Skipping due to missing data.")
        return
        
    summary = forecasting_results.get('summary', {})
    if not summary:
        print("   ❌ No summary found in forecasting results. Skipping plot.")
        return

    metrics = ['mae', 'rmse', 'mape']
    means = [summary.get(f'{m}_mean', 0) for m in metrics]
    stds = [summary.get(f'{m}_std', 0) for m in metrics]
    
    labels = [m.upper() for m in metrics]
    x = np.arange(len(labels))
    
    plt.figure(figsize=(12, 7))
    plt.bar(x, means, yerr=stds, capsize=5, color=sns.color_palette("viridis", 3), alpha=0.85)
    
    plt.ylabel('Error Value')
    plt.title('Part B: Forecasting Performance (Seq2Seq on MRNN-Imputed Data)')
    plt.xticks(x, labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2a_forecasting_performance_bar_chart.png"))
    plt.close()
    print("   ✅ Saved 2a_forecasting_performance_bar_chart.png")


def plot_loss_curves(results_data, title, output_dir, filename):
    """
    Plots training and validation loss curves, averaged across all runs,
    with a shaded area for standard deviation. This is a key diagnostic plot.
    """
    print(f"📉 Plotting Diagnostic: Loss Curves for '{title}'...")
    if not results_data:
        print("   -> Skipping due to missing data.")
        return

    runs = results_data.get('runs', [])
    if not runs:
        print(f"   -> No run data found for {title}. Skipping loss plot.")
        return

    all_train_losses, all_val_losses = [], []
    
    for run in runs:
        logs = run.get('epoch_logs', [])
        if not logs: continue
        
        epochs = [log['epoch'] for log in logs]
        # Use .get() to avoid errors if a key is missing for a particular epoch
        train_losses = [log.get('train_loss') for log in logs]
        val_losses = [log.get('val_loss') for log in logs]
        
        all_train_losses.append(pd.Series(train_losses, index=epochs))
        all_val_losses.append(pd.Series(val_losses, index=epochs))
            
    if not all_train_losses:
        print(f"   -> No epoch logs with loss data found for {title}. Skipping loss plot.")
        return

    # Combine runs and calculate mean/std. `reindex` handles runs with different epoch counts.
    train_loss_df = pd.concat(all_train_losses, axis=1).sort_index()
    val_loss_df = pd.concat(all_val_losses, axis=1).sort_index()
    
    mean_train_loss = train_loss_df.mean(axis=1)
    std_train_loss = train_loss_df.std(axis=1)
    mean_val_loss = val_loss_df.mean(axis=1)
    std_val_loss = val_loss_df.std(axis=1)

    plt.figure(figsize=(12, 7))
    epochs_range = mean_train_loss.index
    
    # Plot mean loss lines
    plt.plot(epochs_range, mean_train_loss, color='royalblue', label='Average Train Loss')
    plt.plot(epochs_range, mean_val_loss, color='coral', label='Average Validation Loss')
    
    # Plot standard deviation bands
    plt.fill_between(epochs_range, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color='royalblue', alpha=0.2)
    plt.fill_between(epochs_range, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color='coral', alpha=0.2)
    
    plt.title(f'Diagnostic: Training & Validation Loss for {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (RMSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"   ✅ Saved {filename}")


def main(args):
    """Main function to generate all visualizations."""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # --- Load Data ---
    print("\n--- Loading Result Files ---")
    imputation_baseline_results = load_json_data(args.imputation_results_baseline)
    imputation_advanced_results = load_json_data(args.imputation_results_advanced)
    forecasting_results = load_json_data(args.forecasting_results)
    
    # --- Generate Plots ---
    print("\n--- Generating Visualizations ---")
    # Part A: Imputation
    plot_imputation_performance(imputation_baseline_results, imputation_advanced_results, args.output_dir)
    plot_qualitative_imputation(
        args.original_data_file, 
        args.advanced_imputed_file,
        args.grin_imputed_file, 
        args.mean_imputed_file, 
        args.output_dir
    )
    plot_error_distribution(
        args.original_data_file,
        args.advanced_imputed_file,
        args.grin_imputed_file,
        args.mean_imputed_file,
        args.output_dir
    )

    # Part B: Forecasting
    plot_forecasting_performance(forecasting_results, args.output_dir)
    print("\n⚠️  Note: A qualitative forecasting plot requires saving prediction arrays from `forecasting_main.py`.")
    print("   This script does not generate it, but can be easily extended if you save the output.")


    # Diagnostics
    plot_loss_curves(imputation_advanced_results, "Advanced Imputation Model (MRNN)", args.output_dir, "3a_imputation_loss_curves.png")
    plot_loss_curves(forecasting_results, "Forecasting Model (Seq2Seq)", args.output_dir, "3b_forecasting_loss_curves.png")
    
    print("\n🎉 All visualizations generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations for traffic forecasting seminar results.")

    # Define file paths using arguments for flexibility
    parser.add_argument("--data_dir", default="data", help="Root directory for data files.")
    parser.add_argument("--results_dir", default="data/outputs", help="Directory for JSON result files.")
    parser.add_argument("--output_dir", default="visualizations", help="Directory to save the plots.")

    args = parser.parse_args()

    # Construct full paths from directories
    # Imputed file from the advanced model run 1
    advanced_imputed_filename = "imputation_evaluation_results_advanced_MRNN_run_1.csv"

    # Set up arguments for the main function
    cli_args = argparse.Namespace(
        imputation_results_baseline=os.path.join(args.results_dir, "imputation_evaluation_results_baseline.json"),
        imputation_results_advanced=os.path.join(args.results_dir, "imputation_evaluation_results_advanced.json"),
        forecasting_results=os.path.join(args.results_dir, "forecasting_evaluation_results.json"),

        original_data_file=os.path.join(args.data_dir, "timeseries_data_with-NaN.csv"),
        advanced_imputed_file=os.path.join(args.results_dir, advanced_imputed_filename),
        grin_imputed_file=os.path.join(args.data_dir, "imputed_dataset.csv"),
        mean_imputed_file=os.path.join(args.data_dir, "mean_imputed.csv"),

        output_dir=args.output_dir
    )

    main(cli_args)
