import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

@dataclass
class InventoryMetrics:
    epoch: int
    batch_id: int
    revenue: float
    holding_cost: float
    total_profit: float

class InventoryLogger:
    def __init__(self):
        self.metrics: List[InventoryMetrics] = []
        
    def log_batch(self, epoch: int, batch_id: int, revenue: float, holding_cost: float):
        total_profit = revenue - holding_cost
        self.metrics.append(InventoryMetrics(
            epoch=epoch,
            batch_id=batch_id,
            revenue=revenue,
            holding_cost=holding_cost,
            total_profit=total_profit
        ))
    
    def get_metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame([vars(m) for m in self.metrics])
    
    def plot_metrics(self):
        df = self.get_metrics_df()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot revenue and holding costs
        df.plot(x='epoch', y=['revenue', 'holding_cost'], ax=ax1)
        ax1.set_title('Revenue and Holding Costs per Epoch')
        ax1.set_ylabel('Cost')
        
        # Plot total profit
        df.plot(x='epoch', y='total_profit', ax=ax2)
        ax2.set_title('Total Profit per Epoch')
        ax2.set_ylabel('Profit')
        
        plt.tight_layout()
        return fig

def modify_trainer_for_logging(trainer: Trainer):
    """
    Modifies the trainer class to include logging functionality.
    """
    original_train_step = trainer.train_step
    logger = InventoryLogger()
    
    def train_step_with_logging(self, *args, **kwargs):
        # Get the original loss and outputs
        loss, outputs = original_train_step(self, *args, **kwargs)
        
        # Extract revenue and holding costs from outputs
        # Note: You'll need to modify this based on your actual output structure
        revenue = outputs.get('revenue', 0.0)
        holding_cost = outputs.get('holding_cost', 0.0)
        
        # Log the metrics
        logger.log_batch(
            epoch=self.current_epoch,
            batch_id=self.current_batch,
            revenue=revenue,
            holding_cost=holding_cost
        )
        
        return loss, outputs
    
    # Replace the original train_step with our logging version
    trainer.train_step = train_step_with_logging.__get__(trainer)
    trainer.logger = logger
    
    return trainer