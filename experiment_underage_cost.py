import yaml
import pandas as pd
from trainer import *
from tqdm import tqdm
import time

def run_single_config(config_setting_file, config_hyperparams_file, underage_cost=None):
    """Run a single configuration and return results"""
    
    # Load configs
    with open(config_setting_file, 'r') as file:
        config_setting = yaml.safe_load(file)

    with open(config_hyperparams_file, 'r') as file:
        config_hyperparams = yaml.safe_load(file)

    # Set underage cost in store params if provided
    if underage_cost is not None:
        # Update the underage cost range to be [0.7*p, 1.3*p] as per paper
        config_setting['store_params']['underage_cost'] = {
            'sample_across_stores': False,
            'vary_across_samples': True,
            'expand': False,
            'range': [0.7 * underage_cost, 1.3 * underage_cost]
        }

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

    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset_creator = DatasetCreator()

    # Create datasets
    if sample_data_params['split_by_period']:
        scenario = Scenario(
            periods=None,
            problem_params=problem_params,
            store_params=store_params,
            warehouse_params=warehouse_params,
            echelon_params=echelon_params,
            num_samples=params_by_dataset['train']['n_samples'],
            observation_params=observation_params,
            seeds=seeds
        )
        
        train_dataset, dev_dataset, test_dataset = dataset_creator.create_datasets(
            scenario,
            split=True,
            by_period=True,
            periods_for_split=[sample_data_params[k] for k in ['train_periods', 'dev_periods', 'test_periods']],
        )
    else:
        scenario = Scenario(
            periods=params_by_dataset['train']['periods'],
            problem_params=problem_params,
            store_params=store_params,
            warehouse_params=warehouse_params,
            echelon_params=echelon_params,
            num_samples=params_by_dataset['train']['n_samples'] + params_by_dataset['dev']['n_samples'],
            observation_params=observation_params,
            seeds=seeds
        )

        train_dataset, dev_dataset = dataset_creator.create_datasets(
            scenario, split=True, by_sample_indexes=True, 
            sample_index_for_split=params_by_dataset['dev']['n_samples']
        )

        scenario = Scenario(
            params_by_dataset['test']['periods'],
            problem_params,
            store_params,
            warehouse_params,
            echelon_params,
            params_by_dataset['test']['n_samples'],
            observation_params,
            test_seeds
        )

        test_dataset = dataset_creator.create_datasets(scenario, split=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=params_by_dataset['train']['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=params_by_dataset['dev']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params_by_dataset['test']['batch_size'], shuffle=False)
    data_loaders = {'train': train_loader, 'dev': dev_loader, 'test': test_loader}

    # Setup model
    neural_net_creator = NeuralNetworkCreator
    model = neural_net_creator().create_neural_network(scenario, nn_params, device=device)
    
    loss_function = PolicyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['learning_rate'])
    
    simulator = Simulator(device=device)
    trainer = Trainer(device=device, experiment_label=f"underage_cost_{underage_cost}")

    # Setup save directories
    trainer_params['base_dir'] = 'saved_models'
    trainer_params['save_model_folders'] = [trainer.get_year_month_day(), nn_params['name']]
    trainer_params['save_model_filename'] = trainer.get_time_stamp()

    if trainer_params['load_previous_model']:
        print(f'Loading model from {trainer_params["load_model_path"]}')
        model, optimizer = trainer.load_model(model, optimizer, trainer_params['load_model_path'])

    # Train
    trainer.train(
        trainer_params['epochs'],
        loss_function, simulator,
        model,
        data_loaders,
        optimizer,
        problem_params,
        observation_params,
        params_by_dataset,
        trainer_params
    )

    # Test
    test_metrics = trainer.test(
        loss_function,
        simulator,
        model,
        data_loaders,
        optimizer,
        problem_params,
        observation_params,
        params_by_dataset,
        trainer_params,
        discrete_allocation=store_params['demand']['distribution'] == 'poisson',
    )

    return {
        'config_name': config_hyperparams_file,
        'underage_cost': underage_cost,
        'test_metrics': test_metrics,
        'model': model
    }

def main():
    # List all config files
    config_setting_file = 'config_files/settings/one_store_real_data_lost_demand.yml'
    config_files = [
        'config_files/policies_and_hyperparams/cyclic_base_stock_quarter.yml',
        'config_files/policies_and_hyperparams/cyclic_capped_base_stock_quarter.yml',
        'config_files/policies_and_hyperparams/cyclic_capped_base_stock_year.yml',
        # 'config_files/policies_and_hyperparams/base_stock.yml',
        # 'config_files/policies_and_hyperparams/vanilla_one_store.yml',
        # 'config_files/policies_and_hyperparams/just_in_time.yml',
        # 'config_files/policies_and_hyperparams/learn_base_stock.yml',
        # 'config_files/policies_and_hyperparams/cyclic_base_stock.yml'
    ]

    # Define underage costs to test (p̂ values from the paper)
    underage_costs = [2, 3, 4, 6, 9, 13, 19]

    # Run all configs with progress bar
    results = []
    for cost in tqdm(underage_costs, desc="Outer Loop: Testing underage costs"):
        for config_file in tqdm(config_files, desc=f"Inner Loop: Running policies (p={cost})", leave=False):
            print(f"\nRunning configuration: {config_file} with underage cost {cost}")
            result = run_single_config(config_setting_file, config_file, underage_cost=cost)
            results.append(result)
            print(f"Test metrics: {result['test_metrics']}\n")

    # Print summary grouped by underage cost
    print("\nFinal Results Summary:")
    print("-" * 50)
    for cost in underage_costs:
        print(f"\nUnderage Cost = {cost}")
        print("-" * 30)
        for result in results:
            if result['underage_cost'] == cost:
                print(f"Config: {result['config_name']}")
                print(f"Test Metrics: {result['test_metrics']}")
                print("-" * 20)

if __name__ == "__main__":
    main()