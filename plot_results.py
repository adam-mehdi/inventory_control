import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

class InventoryControlVisualizer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.colors = {
            'cyclic_base_stock_year': '#1b9e77',    # dark teal
            'cyclic_capped_base_stock_year': '#d95f02', # dark orange
            'cyclic_base_stock': '#7570b3',         # darker lavender-blue
            'base_stock': '#e7298a',                # dark pink
            'capped_base_stock': '#66a61e',         # darker green
            'vanilla_one_store': '#e6ab02',         # mustard yellow
            'learn_base_stock': '#a6761d',          # dark brown
            'just_in_time': '#666666'               # dark gray
        }

        self.names = {
            'cyclic_base_stock_year': 'Cyclic Base Stock Year',
            'cyclic_capped_base_stock_year': 'Cyclic Capped Base Stock Year',
            'cyclic_base_stock': 'Cyclic Base Stock',
            'base_stock': 'Base Stock',
            'capped_base_stock': 'Capped Base Stock',
            'vanilla_one_store': 'Vanilla One Store',
            'learn_base_stock': 'Learn Base Stock',
            'just_in_time': 'Just-in-time'
        }
    
    def extract_param_from_filename(self, filename):
        """Extract both underage_cost and num_samples from filename if present."""
        filename = os.path.basename(filename)
        params = {}
        
        # Extract num_samples
        num_samples_match = re.search(r'num_samples_(\d+)', filename)
        if num_samples_match:
            params['num_samples'] = int(num_samples_match.group(1))
            
        # Extract underage_cost
        underage_match = re.search(r'underage_cost_(\d+)', filename)
        if underage_match:
            params['underage_cost'] = float(underage_match.group(1))
            
        return params

    def load_metrics(self, model_type, phase='test'):
        pattern = os.path.join(self.base_dir, model_type, f'*_metrics_{phase}.json')
        files = glob.glob(pattern)
        data = []
        
        for file in files:
            params = self.extract_param_from_filename(file)
            if not params:
                continue
                
            with open(file, 'r') as f:
                metrics = json.load(f)
                
                if phase in metrics:
                    phase_data = metrics[phase]
                    if isinstance(phase_data, dict):
                        revenue = phase_data.get('revenue', 0)
                        holding_costs = phase_data.get('holding_costs', 0)
                        
                        entry = {
                            'revenue': revenue,
                            'holding_costs': holding_costs,
                            'profit': revenue - holding_costs,
                            **params  # Add both parameters if present
                        }
                        data.append(entry)
                            
        return pd.DataFrame(data)

    def plot_metrics_by_param(self, model_groups, param_name, title):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['profit', 'revenue', 'holding_costs']
        metric_labels = {
            'profit': 'Average Profit',
            'revenue': 'Average Revenue',
            'holding_costs': 'Average Holding Cost'
        }
        
        for ax, metric in zip(axes, metrics):
            for model in model_groups:
                data = self.load_metrics(model)
                if not data.empty and param_name in data.columns:
                    data_grouped = data.groupby(param_name)[metric].mean().reset_index()
                    data_grouped = data_grouped.sort_values(param_name)
                    
                    ax.plot(data_grouped[param_name], data_grouped[metric],
                            label=self.names[model],
                            color=self.colors[model],
                            marker='o',
                            linewidth=2)
            
            ax.set_title(metric_labels[metric])
            x_label = 'Average Unit Underage Cost' if param_name == 'underage_cost' else 'Number of Samples'
            ax.set_xlabel(x_label)
            ax.set_ylabel(metric_labels[metric])
            
            if metric == 'profit':
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set x-axis to log scale for num_samples
            if param_name == 'num_samples':
                ax.set_xscale('log')
                
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title + " Per Store-Product Pair", y=1.05)
        plt.tight_layout()
        return fig

    def style_plot(self):
        plt.style.use('seaborn')
        sns.set_palette("deep")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['figure.titlesize'] = 16

# Example usage
if __name__ == "__main__":
    base_dir = "results/2024_12_08"
    visualizer = InventoryControlVisualizer(base_dir)
    visualizer.style_plot()
    
    # Define model groups
    base_stock_models = ['cyclic_base_stock_year', 'cyclic_capped_base_stock_year',  
                        'base_stock', 'capped_base_stock']
    learning_models = ['vanilla_one_store', 'learn_base_stock']
    all_models = base_stock_models + learning_models + ['just_in_time']
    
    # Create plots for both parameter types
    configs = [
        (base_stock_models, 'Base Stock Models'),
        (learning_models, 'Learning Models'),
        (all_models, 'All Models')
    ]
    
    # Plot underage cost variations
    for models, name in configs:
        fig = visualizer.plot_metrics_by_param(
            models, 
            'underage_cost',
            f'{name} Performance (Varying Underage Cost)'
        )
        fig.savefig(f'plots/{name.lower().replace(" ", "_")}_underage_cost.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
    
    # Plot num_samples variations
    for models, name in configs:
        fig = visualizer.plot_metrics_by_param(
            models,
            'num_samples',
            f'{name} Performance (Varying Number of Samples)'
        )
        fig.savefig(f'plots/{name.lower().replace(" ", "_")}_num_samples.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
