import yaml
import pandas as pd
from trainer import *
from tqdm import tqdm
import time

def run_single_config(config_setting_file, config_hyperparams_file, n_samples=None):
    """Run a single configuration and return results"""
    
    # Load configs
    with open(config_setting_file, 'r') as file:
        config_setting = yaml.safe_load(file)

    with open(config_hyperparams_file, 'r') as file:
        config_hyperparams = yaml.safe_load(file)

    # Modify number of samples if provided
    if n_samples is not None:
        config_setting['params_by_dataset']['train']['n_samples'] = n_samples
        config_setting['params_by_dataset']['dev']['n_samples'] = n_samples
        config_setting['params_by_dataset']['test']['n_samples'] = n_samples
        
        # Adjust batch sizes to be no larger than sample size
        for dataset in ['train', 'dev', 'test']:
            config_setting['params_by_dataset'][dataset]['batch_size'] = min(
                n_samples, 
                config_setting['params_by_dataset'][dataset]['batch_size']
            )

    # Extract parameters from configs
    setting_keys = ('seeds', 'test_seeds', 'problem_params', 'params_by_dataset', 
                   'observation_params', 'store_params', 'warehouse_params', 
                   'echelon_params', 'sample_data_params')
    hyperparams_keys = ('trainer_params', 'optimizer_params', 'nn_params')
    
    (seeds, test_seeds, problem_params, params_by_dataset, observation_params, 
     store_params, warehouse_params, echelon_params, sample_data_params) = [
        config_setting[key] for key in setting_keys
    ]

    trainer_params, optimizer_params, nn_params = [config_hyperparams[key] for key in hyperparams_keys]
    observation_params = DefaultDict(lambda: None, observation_params)

    # Rest of the original function remains the same
    # ... (keep all the existing code for dataset creation, model setup, training, and testing)

    return {
        'config_name': config_hyperparams_file,
        'n_samples': n_samples,
        'test_metrics': test_metrics,
        'model': model
    }

def main():
    # List all config files
    config_setting_file = 'config_files/settings/one_store_real_data_lost_demand.yml'
    config_files = [
        'config_files/policies_and_hyperparams/capped_base_stock.yml',
        'config_files/policies_and_hyperparams/base_stock.yml',
        'config_files/policies_and_hyperparams/vanilla_one_store.yml',
        'config_files/policies_and_hyperparams/just_in_time.yml',
        'config_files/policies_and_hyperparams/learn_base_stock.yml',
        'config_files/policies_and_hyperparams/cyclic_base_stock.yml'
    ]

    # Define sample sizes to test (geometric progression)
    sample_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768]

    # Run all configs with progress bar
    results = []
    for samples in tqdm(sample_sizes, desc="Outer Loop: Testing sample sizes"):
        for config_file in tqdm(config_files, desc=f"Inner Loop: Running policies (n={samples})", leave=False):
            print(f"\nRunning configuration: {config_file} with {samples} samples")
            result = run_single_config(config_setting_file, config_file, n_samples=samples)
            results.append(result)
            print(f"Test metrics: {result['test_metrics']}\n")

    # Print summary grouped by sample size
    print("\nFinal Results Summary:")
    print("=" * 50)
    for samples in sample_sizes:
        print(f"\nSample Size = {samples}")
        print("-" * 30)
        for result in results:
            if result['n_samples'] == samples:
                print(f"Config: {result['config_name']}")
                print(f"Test Metrics: {result['test_metrics']}")
                print("-" * 20)

    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'sample_size': r['n_samples'],
            'config': r['config_name'],
            **r['test_metrics']
        }
        for r in results
    ])
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_df.to_csv(f'results_sample_size_experiment_{timestamp}.csv', index=False)

if __name__ == "__main__":
    main()