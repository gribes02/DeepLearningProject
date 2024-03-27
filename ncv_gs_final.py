import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import mean_absolute_error
import torch
import numpy as np
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CNN(nn.Module):
    def __init__(self, nr_conv_layers = 3, 
                       nr_ff_layers = 1, 
                       out_channels = [50, 70, 70], 
                       ff_output_sizes = [25], 
                       kernel_sizes = [5, 3, 3]):
        super(CNN, self).__init__()
        
        # Storage for convolutional and feedforward layers
        self.conv_layers = nn.ModuleList()
        self.fnn_layers = nn.ModuleList()
        
        # Dimension of input images (256x256)
        conv_output_size = 256
                
        # Create and store convolutional layers
        for i in range(nr_conv_layers):
            conv_output = out_channels[i]
            conv_size = kernel_sizes[i]
            if i == 0: 
                nr_conv_input = 1
            else: 
                nr_conv_input = out_channels[i-1]
                
            conv_layer = nn.Conv2d(nr_conv_input, conv_output, conv_size)
            self.conv_layers.append(conv_layer)
            
            # Calculate size of output
            conv_output_size = conv_output_size - (conv_size // 2)*2
            conv_output_size = conv_output_size / 2

        conv_output_size = int(conv_output_size)
        
        # Local Response Normalization
        self.norm = nn.LocalResponseNorm(size=4, alpha=0.001 / 9.0, beta=0.75, k=1.0)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Store and create feedforward layers
        for i in range(nr_ff_layers):
            if i == 0:
                input_features = out_channels[-1] * conv_output_size**2
            else:
                input_features = ff_output_sizes[i-1]
            fnn_layer = nn.Linear(input_features, ff_output_sizes[i])
            self.fnn_layers.append(fnn_layer)

    def forward(self, x, apply_softmax = False):
        """Forward step"""

        # Pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = self.norm(self.pool(F.relu(conv_layer(x))))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Pass through feedforward layers
        for i, fnn_layer in enumerate(self.fnn_layers):
            if i < len(self.fnn_layers) - 1:
                x = F.relu(fnn_layer(x))
            else:
                x = fnn_layer(x)
                
        # Apply Softmax at evaluation
        if apply_softmax:
            x = F.softmax(x, dim=1)
        return x
    
    def reset_weights(self):
        """Reset model weights"""

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
class HyperParamTuner():
    def __init__(self, num_epochs, batch_size, init_lr, nr_outer_folds, nr_inner_folds, configs):
        self.num_epochs     = num_epochs
        self.batch_size     = batch_size
        self.learning_rate  = init_lr

        self.nr_outer_folds = nr_outer_folds
        self.nr_inner_folds = nr_inner_folds

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.param_grid = configs

        # self.inner_load_path = '/kaggle/working/best_inner_model.pt'
        # self.outer_load_path = '/kaggle/working/best_outer_model.pt'
        # self.log_file        = '/kaggle/working/log_file.txt'

        self.inner_load_path = 'best_inner_model.pt'
        self.outer_load_path = 'best_outer_model.pt'
        self.log_file        = 'log_file.txt'

        self.best_inner_config = None
        self.best_model        = None
        self.best_accuracy     = 0.0
        self.final_mae         = 0.0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])

        self.dataset = ImageFolder(root='data/malimg_paper_dataset_imgs', transform=transform)
        self.targets = self.dataset.targets

    def write_to_log(self, message):
        """Write messages to the log file."""
        with open(self.log_file, 'a') as log_file:
            log_file.write(message + "\n")

    def load_model(self, load_path):
        """Load stored models"""
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            print(f"Model loaded from {load_path}")
        else:
            print(f"No model found at {load_path}. Please check the path and try again.")

    def evaluate_model(self, test_loader):
        """Evaluate model on evaluation set and return
           accuracy and mae"""
        total_val   = 0
        correct_val = 0
        val_predictions = []
        val_true_labels = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs, apply_softmax = True)

                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

                val_true_labels.extend(targets.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())
                
        val_accuracy = 100 * correct_val / total_val
        val_mae = mean_absolute_error(val_true_labels, val_predictions)

        return val_accuracy, val_mae
    
    def train_model(self, train_set, val_set):        
        """Train model on training set and return accuracy and
           mae of best epoch"""
        train_loader = DataLoader(train_set, batch_size=self.batch_size, 
                                shuffle=True, num_workers=2)
        test_loader = DataLoader(val_set, batch_size=self.batch_size, 
                                shuffle=False, num_workers=2)
                    
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=6, verbose=True)

        best_val_accuarcy = 0.0
        final_mae         = float(np.inf)
        epochs_no_improve = 0
            
        for epoch in range(0, self.num_epochs):
            
            self.model.train()
            for i, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = self.model(inputs)
                            
                # Compute loss
                loss = self.loss_function(outputs, targets)
                
                # Perform backward pass
                loss.backward()
                
                # Perform optimization
                optimizer.step()

            val_accuracy, val_mae = self.evaluate_model(test_loader)
            scheduler.step(val_accuracy)

            print(f'Epoch {epoch + 1}, Val Accuracy: {val_accuracy:.2f}, Val MAE: {val_mae:.2f}')

            if val_accuracy > best_val_accuarcy:
                best_val_accuarcy = val_accuracy
                final_mae = val_mae
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 10:
                    print("Early stopping triggered")
                    break

        return best_val_accuarcy, final_mae

    def outerFold(self):
        """Execute outer fold of nested cross validation"""
        outer_kfold = StratifiedKFold(n_splits=self.nr_outer_folds, shuffle=True)

        outer_fold_accuracies = []
        outer_fold_maes       = []
        best_accuracy         = 0

        for outer_fold, (outer_train_ids, outer_test_ids) in enumerate(outer_kfold.split(self.dataset, self.targets)):
            print(f'\nOUTER FOLD: {outer_fold}')
            print('------------------------------------------')
            print('------------------------------------------')
            
            self.write_to_log(f'outer fold {outer_fold+1}/{self.nr_outer_folds}')
            
            outer_training_set = Subset(self.dataset, outer_train_ids) 
            outer_training_targets = [self.targets[i] for i in outer_train_ids]

            outer_test_set = Subset(self.dataset, outer_test_ids)

            best_inner_val_accuracy = 0
            best_inner_val_mae      = float('inf')

            for config in self.param_grid:
                self.model = CNN(config["nr_conv_layers"],
                                 config["nr_ff_layers"],
                                 config["out_channels"],
                                 config["ff_output_sizes"],
                                 config["kernel_size"]).to(self.device)
                
                print("\nModel: \n", self.model)
                
                avg_inner_val_accuracy, avg_inner_val_mae = self.innerFold(outer_training_set, outer_test_set, outer_training_targets, config)

                if avg_inner_val_accuracy > best_inner_val_accuracy:
                    print("\nThis is the best inner model so far!")
                    best_inner_val_accuracy = avg_inner_val_accuracy
                    best_inner_val_mae      = avg_inner_val_mae
                    best_inner_model_state  = self.model.state_dict()
                    self.best_inner_config  = config
                    torch.save(best_inner_model_state, self.inner_load_path)

            self.write_to_log(
                f"best best_inner_val_accuracy {best_inner_val_accuracy}, best_inner_val_mae {best_inner_val_mae}")

            print("\nTesting best inner model on outer fold")

            self.load_model(self.inner_load_path)
            test_accuracy, test_mae = self.train_model(outer_training_set, outer_test_set)

            print(f"Test Accuracy for Outer Fold {outer_fold + 1}: {test_accuracy:.2f}%")
            print(f"Test MAE for Outer Fold {outer_fold + 1}: {test_mae:.2f}")

            self.write_to_log(f"Best inner model: \n{self.model}")
            self.write_to_log(
                f"Test Accuracy for Outer Fold {outer_fold + 1}: {test_accuracy:.2f}%, Test MAE for Outer Fold {outer_fold + 1}: {test_mae:.2f}",
                typ="statistics")

            outer_fold_accuracies.append(test_accuracy)
            outer_fold_maes.append(test_mae)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_mae      = test_mae
                best_outer_model_state = self.model.state_dict()

                torch.save(best_outer_model_state, self.outer_load_path)

                print(f"saving outer model with accuracy {best_accuracy}, {best_mae}")
            
        outer_fold_accuracy_std = np.std(outer_fold_accuracies)
        outer_fold_mae_std      = np.std(outer_fold_maes)

        self.write_to_log(
            f"Standard deviation(accuracy): {outer_fold_accuracy_std}, outer_fold_mae_std: {outer_fold_mae_std}",
            typ="statistics")
        
        print(f"\nStandard deviation accuracy outer fold: {outer_fold_accuracy_std}, standard deviation mae outer fold: {outer_fold_mae_std}")
        

    def innerFold(self, outer_training_set, outer_test_set, outer_training_targets, config):
        """Execute inner fold of nested cross validation and return
           average accuracy and mae over all inner folds"""
        inner_kfold = StratifiedKFold(n_splits=self.nr_inner_folds, shuffle=True)

        inner_accuracies = []
        inner_maes       = []

        for inner_fold, (inner_train_ids, inner_test_ids) in enumerate(inner_kfold.split(outer_training_set, outer_training_targets)):
            print(f'\nINNER FOLD {inner_fold}')
            print('--------------------------------')

            self.write_to_log(f'inner fold {inner_fold+1}/{self.nr_inner_folds}')

            self.model.reset_weights()
            
            inner_train_set = Subset(outer_training_set, inner_train_ids)
            inner_test_set = Subset(outer_training_set, inner_test_ids)
            val_accuracy, val_mae = self.train_model(inner_train_set, inner_test_set)

            inner_accuracies.append(val_accuracy)
            inner_maes.append(val_mae)

        avg_inner_val_accuracy = np.mean(inner_accuracies)
        avg_inner_val_mae = np.mean(inner_maes)
        
        self.write_to_log(f"For configuration {config}")
        self.write_to_log(f"Avg accuracy: {avg_inner_val_accuracy:.2f}%, avg mae: {avg_inner_val_mae:.2f} \n")
        
        print(f'\nAvg inner accuracy: {avg_inner_val_accuracy} \nAvg inner mae: {avg_inner_val_mae}')

        return avg_inner_val_accuracy, avg_inner_val_mae

            
def get_configs(param_grid):
    """Get valid configurations from search space"""
    configs = []

    for nr_conv_layer in param_grid["nr_conv_layers"]:
        for nr_ff_layer in param_grid["nr_ff_layers"]:
            for out_channel in param_grid["out_channels"]:
                for ff_output_size in param_grid["ff_output_sizes"]:
                    for kernel_size in param_grid["kernel_sizes"]:
                        if nr_conv_layer == len(out_channel) and nr_ff_layer == len(ff_output_size) and nr_conv_layer == len(kernel_size):
                            dct = {"nr_conv_layers": nr_conv_layer,
                                   "nr_ff_layers": nr_ff_layer,
                                   "out_channels": out_channel,
                                   "ff_output_sizes": ff_output_size,
                                   "kernel_size": kernel_size}

                            configs.append(dct)
    return configs

if __name__ == '__main__':
    # Hyper-parameters 
    num_epochs = 30
    batch_size = 200
    learning_rate = 0.001
    nr_outer_folds = 3
    nr_inner_folds = 5

#     param_grid = {
#         'nr_conv_layers': [4],
#         'nr_ff_layers': [1, 2, 3],
#         'out_channels':  [[20, 30, 40, 50], [50, 70, 70, 90]],
#         'ff_output_sizes': [[25], [50, 25], [70, 25], [50, 70, 25], [50, 50, 25]],
#         'kernel_sizes': [[5, 3, 3, 3], [9, 5, 5, 3], [13, 7, 3, 3]]
#     }

    param_grid = {
        'nr_conv_layers': [4],
        'nr_ff_layers': [1],
        'out_channels':  [[20, 30, 40, 50]],
        'ff_output_sizes': [[25]],
        'kernel_sizes': [[5, 3, 3, 3]]
    }

    configs = get_configs(param_grid)

    hyper_param_tuner = HyperParamTuner(num_epochs,
                                        batch_size,
                                        learning_rate,
                                        nr_outer_folds,
                                        nr_inner_folds,
                                        configs)

    hyper_param_tuner.outerFold()