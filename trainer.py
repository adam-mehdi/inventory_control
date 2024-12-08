from shared_imports import *
from environment import *
from loss_functions import *
import json

class Trainer():
    """
    Trainer class
    """

    def __init__(self,  device='cpu', underage_cost=None):
        
        self.all_train_losses = []
        self.all_dev_losses = []
        self.all_test_losses = [] 
        self.device = device
        self.time_stamp = self.get_time_stamp()
        self.best_performance_data = {'train_loss': np.inf, 'dev_loss': np.inf, 'last_epoch_saved': -1000, 'model_params_to_save': None}
        self.underage_cost = underage_cost
    
    def reset(self):
        """
        Reset the losses
        """

        self.all_train_losses = []
        self.all_dev_losses = []
        self.all_test_losses = []

    def train(self, epochs, loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params):
        """
        Train a parameterized policy

        Parameters:
        ----------
        epochs: int
            Number of epochs to train the policy
        loss_function: LossFunction
            Loss function to use for training.
            In our case, we will use PolicyLoss, that directly calculates the loss as sum of the rewards/costs
        simulator: Simulator(gym.Env)
            Differentiable Gym environment to use for simulating.
            In our experiments, this will simulate a specific inventory problem for a number of periods
        model: nn.Module
            Neural network model to train
        data_loaders: dict
            Dictionary containing the DataLoader objects for train, dev and test datasets
        optimizer: torch.optim
            Optimizer to use for training the model.
            In our experiments we use Adam optimizer
        problem_params: dict
            Dictionary containing the problem parameters, specifying the number of warehouses, stores, whether demand is lost
            and whether to maximize profit or minimize underage+overage costs
        observation_params: dict
            Dictionary containing the observation parameters, specifying which features to include in the observations
        params_by_dataset: dict
            Dictionary containing the parameters for each dataset, such as the number of periods, number of samples, batch size
        trainer_params: dict
            Dictionary containing the parameters for the trainer, such as the number of epochs between saving the model, the base directory
            where to save the model, the filename for the model, whether to save the model, the number of epochs between saving the model
            and the metric to use for choosing the best model
        """

        self.all_train_revenues = []
        self.all_train_holding_costs = []
        self.all_dev_revenues = []
        self.all_dev_holding_costs = []

        for epoch in range(epochs): # Make multiple passes through the dataset
            
            # Do one epoch of training, including updating the model parameters
            train_metrics = self.do_one_epoch(
                optimizer, 
                data_loaders['train'], 
                loss_function, 
                simulator, 
                model, 
                params_by_dataset['train']['periods'], 
                problem_params, 
                observation_params, 
                train=True, 
                ignore_periods=params_by_dataset['train']['ignore_periods']
            )
            average_train_loss, average_train_loss_to_report, train_revenue, train_holding_costs = train_metrics
            
            self.all_train_losses.append(average_train_loss_to_report)
            self.all_train_revenues.append(train_revenue)
            self.all_train_holding_costs.append(train_holding_costs)

            if epoch % trainer_params['do_dev_every_n_epochs'] == 0:
                dev_metrics = self.do_one_epoch(
                    optimizer, 
                    data_loaders['dev'], 
                    loss_function, 
                    simulator, 
                    model, 
                    params_by_dataset['dev']['periods'], 
                    problem_params, 
                    observation_params, 
                    train=False, 
                    ignore_periods=params_by_dataset['dev']['ignore_periods']
                    )
                average_dev_loss, average_dev_loss_to_report, dev_revenue, dev_holding_costs = dev_metrics
            
                self.all_dev_losses.append(average_dev_loss_to_report)
                self.all_dev_revenues.append(dev_revenue)
                self.all_dev_holding_costs.append(dev_holding_costs)

                # Check if the current model is the best model so far, and save the model parameters if so.
                # Save the model if specified in the trainer_params
                self.update_best_params_and_save(epoch, average_train_loss_to_report, average_dev_loss_to_report, trainer_params, model, optimizer)
                
            else:
                average_dev_loss, average_dev_loss_to_report = 0, 0
                self.all_dev_losses.append(self.all_dev_losses[-1])
                self.all_dev_revenues.append(self.all_dev_revenues[-1])
                self.all_dev_holding_costs.append(self.all_dev_holding_costs[-1])

            # Print epoch number and average per-period loss every 10 epochs
            if epoch % trainer_params['print_results_every_n_epochs'] == 0:
                print(f'Epoch: {epoch + 1}')
                print(f'Train - Loss: {average_train_loss_to_report:.2f}, Revenue: {train_revenue:.2f}, Holding Costs: {train_holding_costs:.2f}')
                if epoch % trainer_params['do_dev_every_n_epochs'] == 0:
                    print(f'Dev - Loss: {average_dev_loss_to_report:.2f}, Revenue: {dev_revenue:.2f}, Holding Costs: {dev_holding_costs:.2f}')
        
        metrics = {
            'train': {
                'losses': self.all_train_losses,
                'revenues': self.all_train_revenues,
                'holding_costs': self.all_train_holding_costs
            },
            'dev': {
                'losses': self.all_dev_losses,
                'revenues': self.all_dev_revenues,
                'holding_costs': self.all_dev_holding_costs
            }
        }
        self.save_metrics_to_json(trainer_params, metrics)

    
    def test(self, loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params, discrete_allocation=False):

        if model.trainable and self.best_performance_data['model_params_to_save'] is not None:
            # Load the parameter weights that gave the best performance on the specified dataset
            model.load_state_dict(self.best_performance_data['model_params_to_save'])

        test_metrics = self.do_one_epoch(
                optimizer, 
                data_loaders['test'], 
                loss_function, 
                simulator, 
                model, 
                params_by_dataset['test']['periods'], 
                problem_params, 
                observation_params, 
                train=True, 
                ignore_periods=params_by_dataset['test']['ignore_periods'],
                discrete_allocation=discrete_allocation
            )
        

        # Save test metrics
        metrics = {
            'test': {
                'loss': float(test_metrics[0]),
                'revenue': float(test_metrics[2]),
                'holding_costs': float(test_metrics[3])
            }
        }
        self.save_metrics_to_json(trainer_params, metrics, is_training=False)
        
        return test_metrics

    def do_one_epoch(self, optimizer, data_loader, loss_function, simulator, model, periods, problem_params, observation_params, train=True, ignore_periods=0, discrete_allocation=False):
        """
        Do one epoch of training or testing
        """
        
        epoch_loss = 0
        epoch_loss_to_report = 0  # Loss ignoring the first 'ignore_periods' periods
        epoch_revenue = 0
        epoch_holding_costs = 0
        total_samples = len(data_loader.dataset)
        periods_tracking_loss = periods - ignore_periods  # Number of periods for which we report the loss

        for i, data_batch in enumerate(data_loader):  # Loop through batches of data
            data_batch = self.move_batch_to_device(data_batch)
            
            if train:
                # Zero-out the gradient
                optimizer.zero_grad()

            # Forward pass
            total_reward, reward_to_report, revenue, holding_costs = self.simulate_batch(
                loss_function, simulator, model, periods, problem_params, data_batch, observation_params, ignore_periods, discrete_allocation
                )
            epoch_loss += total_reward.item()  # Rewards from period 0
            epoch_loss_to_report += reward_to_report.item()  # Rewards from period ignore_periods onwards
            epoch_revenue += revenue.item()
            epoch_holding_costs += holding_costs.item()

            
            mean_loss = total_reward/(len(data_batch['demands'])*periods*problem_params['n_stores'])
            
            # Backward pass (to calculate gradient) and take gradient step
            if train and model.trainable:
                mean_loss.backward()
                optimizer.step()
        
        # return epoch_loss/(total_samples*periods*problem_params['n_stores']), epoch_loss_to_report/(total_samples*periods_tracking_loss*problem_params['n_stores'])
            # Calculate normalized metrics
        norm_factor = total_samples * periods_tracking_loss * problem_params['n_stores']
        metrics = {
            'loss': epoch_loss/(total_samples*periods*problem_params['n_stores']),
            'loss_report': epoch_loss_to_report/norm_factor,
            'revenue': epoch_revenue/norm_factor,
            'holding_costs': epoch_holding_costs/norm_factor
        }
        
        return metrics['loss'], metrics['loss_report'], metrics['revenue'], metrics['holding_costs']

    
    def simulate_batch(self, loss_function, simulator, model, periods, problem_params, data_batch, observation_params, ignore_periods=0, discrete_allocation=False):
        """
        Simulate for an entire batch of data, across the specified number of periods
        """

        # Initialize reward across batch
        batch_reward = 0
        reward_to_report = 0
        revenue_to_report = 0
        holding_costs_to_report = 0


        observation, _ = simulator.reset(periods, problem_params, data_batch, observation_params)
        for t in range(periods):

            # We add internal data to the observation to create non-admissible benchmark policies.
            # No admissible policy should use data stored in _internal_data!
            observation_and_internal_data = {k: v for k, v in observation.items()}
            observation_and_internal_data['internal_data'] = simulator._internal_data

            # Sample action
            action = model(observation_and_internal_data)
            
            if discrete_allocation:  # Round actions to the nearest integer if specified
                action = {key: val.round() for key, val in action.items()}

            observation, reward, terminated, _, _  = simulator.step(action)

            total_reward = loss_function(None, action, reward)

            batch_reward += total_reward
            if t >= ignore_periods:
                reward_to_report += total_reward
                revenue_to_report += observation['_revenue'].sum()
                holding_costs_to_report += observation['_holding_costs'].sum()

            
            if terminated:
                break

        # Return reward
        return batch_reward, reward_to_report, revenue_to_report, holding_costs_to_report

    def save_model(self, epoch, model, optimizer, trainer_params):

        path = self.create_many_folders_if_not_exist_and_return_path(base_dir=trainer_params['base_dir'], 
                                                                     intermediate_folder_strings=trainer_params['save_model_folders']
                                                                     )
        underage_cost = f"_{self.underage_cost}" if self.underage_cost else ""
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_performance_data['model_params_to_save'],
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_train_loss': self.best_performance_data['train_loss'],
                    'best_train_loss': self.best_performance_data['dev_loss'],
                    'best_dev_loss': self.all_train_losses,
                    'all_train_losses': self.all_train_losses,
                    'all_dev_losses': self.all_dev_losses,
                    'all_test_losses': self.all_test_losses,
                    'warehouse_upper_bound': model.warehouse_upper_bound
                    }, 
                    f"{path}/{trainer_params['save_model_filename']}{underage_cost}.pt"
                    )
    
    def create_folder_if_not_exists(self, folder):
        """
        Create a directory in the corresponding file, if it does not already exist
        """

        if not os.path.isdir(folder):
            os.mkdir(folder)
    
    def create_many_folders_if_not_exist_and_return_path(self, base_dir, intermediate_folder_strings):
        """
        Create a directory in the corresponding file for each file in intermediate_folder_strings, if it does not already exist
        """

        path = base_dir
        for string in intermediate_folder_strings:
            path += f"/{string}"
            self.create_folder_if_not_exists(path)
        return path
    
    def update_best_params_and_save(self, epoch, train_loss, dev_loss, trainer_params, model, optimizer):
        """
        Update best model parameters if it achieves best performance so far, and save the model
        """

        data_for_compare = {'train_loss': train_loss, 'dev_loss': dev_loss}
        if data_for_compare[trainer_params['choose_best_model_on']] < self.best_performance_data[trainer_params['choose_best_model_on']]:  
            self.best_performance_data['train_loss'] = train_loss
            self.best_performance_data['dev_loss'] = dev_loss
            if model.trainable:
                self.best_performance_data['model_params_to_save'] = copy.deepcopy(model.state_dict())
            self.best_performance_data['update'] = True

        if trainer_params['save_model'] and model.trainable:
            if self.best_performance_data['last_epoch_saved'] + trainer_params['epochs_between_save'] <= epoch and self.best_performance_data['update']:
                self.best_performance_data['last_epoch_saved'] = epoch
                self.best_performance_data['update'] = False
                self.save_model(epoch, model, optimizer, trainer_params)
    
    def plot_losses(self, ymin=None, ymax=None):
        """
        Plot train and test losses for each epoch
        """

        plt.plot(self.all_train_losses, label='Train loss')
        plt.plot(self.all_dev_losses, label='Dev loss')
        plt.legend()

        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    def move_batch_to_device(self, data_batch):
        """
        Move a batch of data to the device (CPU or GPU)
        """

        return {k: v.to(self.device) for k, v in data_batch.items()}
    
    def load_model(self, model, optimizer, model_path):
        """
        Load a saved model
        """

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.all_train_losses = checkpoint['all_train_losses']
        self.all_dev_losses = checkpoint['all_dev_losses']
        self.all_test_losses = checkpoint['all_test_losses']
        model.warehouse_upper_bound = checkpoint['warehouse_upper_bound']
        return model, optimizer
    
    def get_time_stamp(self):

        return int(datetime.datetime.now().timestamp())
    
    def get_year_month_day(self):
        """"
        Get current date in year_month_day format
        """

        ct = datetime.datetime.now()
        return f"{ct.year}_{ct.month:02d}_{ct.day:02d}"
    
    def save_metrics_to_json(self, trainer_params, metrics_dict, is_training=True):
        """Save metrics to a JSON file."""
        path = self.create_many_folders_if_not_exist_and_return_path(
            base_dir=trainer_params['base_dir'],
            intermediate_folder_strings=trainer_params['save_model_folders']
        )
        
        underage_cost = f"_{self.underage_cost}" if self.underage_cost else ""
        phase = "train" if is_training else "test"
        with open(f"{path}/{trainer_params['save_model_filename']}{underage_cost}_metrics_{phase}.json", 'w') as f:
            json.dump(metrics_dict, f, indent=4)
