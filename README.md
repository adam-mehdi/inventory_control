# Inventory Control with Direct Policy Optimization

This repository contains our (Adam Mehdi, Kevin Qiu, Rene Sultan) class project for ORCSE4529_001_2024_3 Reinforcement Learning (Spring 2024) with Professor Shipra Agrawal. The project extends the implementation of neural inventory control policies from Alvo et al. (2023) with new implementations and experiments.

## Contributions 

### Metric Tracking
The trainer now incorporates a more comprehensive metric tracking system in trainer.py. This includes tracking of holding costs, stockout costs, and total rewards over training epochs, providing better visibility into model performance during training. The metrics are logged throughout training and used for comparative analysis between different model architectures.

### New Inventory Control Policies
Several new policies have been implemented in neural_networks.py. The CyclicBaseStockQuarter implements a 13-week quarterly cycle for base stock levels, while CyclicCappedBaseStockYear and CyclicCappedBaseStockQuarter implement annual and quarterly cycles respectively with both base stock levels and order caps. These policies are designed to capture seasonal patterns at different time scales. The LearnBaseStock policy dynamically learns base stock levels from context and computes orders using these learned levels.

### Actor-Critic Implementation
We implemented an experimental Actor-Critic model in A2C_Inventory.ipynb as a reinforcement learning baseline. The implementation includes a shared feature extractor, multiple actor heads (one per product), and a value function critic. 

This was not included in our final writeup due to training instability. The current implementation exhibits high variance in rewards and unstable learning behavior, making it unsuitable for direct comparison with the other implemented policies. We include it in the repository as documentation of our exploration of conventional RL approaches to this problem.

## Experiments

### Sample Size Impact (experiment_num_samples.py)
This experiment evaluates how different sample sizes affect policy performance. It tests sample sizes ranging from 32 to 32768 across multiple policy types including base stock, capped base stock, and just-in-time policies. The experiment helps understand the data requirements for effective policy learning and the scalability of different approaches.

### Underage Cost Analysis (experiment_underage_cost.py)
This experiment investigates the effect of varying underage costs on policy performance. It tests costs ranging from 2 to 19 units, focusing particularly on cyclic base stock policies with different cycle lengths. The experiment follows the paper's methodology of using underage cost ranges [0.7*p, 1.3*p] where p is the base cost. The results help understand how different policies adapt to varying cost structures.

## Project Structure
```
.
├── neural_networks.py             # Policy implementations
├── trainer.py                     # Training and metrics
├── experiment_num_samples.py      # Sample size experiments
├── experiment_underage_cost.py    # Cost analysis experiments
├── A2C_Inventory.ipynb           # Actor-Critic implementation
└── config_files/                 # Configuration files
```

## Original Implementation

This project builds upon the work from primarily:

```
@article{alvo2023neural,
  title={Neural inventory control in networks via hindsight differentiable policy optimization},
  author={Alvo, Matias and Russo, Daniel and Kanoria, Yash},
  journal={arXiv preprint arXiv:2306.11246},
  year={2023}
}
```

and their [repository](https://github.com/MatiasAlvo/Neural_inventory_control).

