"""
Parameter sensitivity analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class ParameterSensitivityAnalyzer:
    """Analyze model sensitivity to hyperparameters"""

    def __init__(self, model_class, train_func, eval_func):
        self.model_class = model_class
        self.train_func = train_func
        self.eval_func = eval_func

    def analyze_visual_concepts(self, K_values, base_config):
        """
        Analyze sensitivity to number of visual concepts

        Args:
            K_values: List of K values to test
            base_config: Base configuration dictionary

        Returns:
            Dictionary with results for each K
        """
        results = {}

        for K in K_values:
            print(f"Testing K={K}...")

            config = base_config.copy()
            config['num_visual_concepts'] = K

            model = self.model_class(**config)
            self.train_func(model, config)
            metrics = self.eval_func(model)

            results[K] = metrics

        return results

    def analyze_learning_rate(self, lr_values, base_config):
        """Analyze sensitivity to learning rate"""
        results = {}

        for lr in lr_values:
            print(f"Testing learning_rate={lr}...")

            config = base_config.copy()
            config['learning_rate'] = lr

            model = self.model_class(**config)
            self.train_func(model, config)
            metrics = self.eval_func(model)

            results[lr] = metrics

        return results

    def analyze_embedding_dimension(self, dim_values, base_config):
        """Analyze sensitivity to embedding dimension"""
        results = {}

        for dim in dim_values:
            print(f"Testing embedding_dim={dim}...")

            config = base_config.copy()
            config['structural_dim'] = dim

            model = self.model_class(**config)
            self.train_func(model, config)
            metrics = self.eval_func(model)

            results[dim] = metrics

        return results

    def analyze_negative_sampling(self, num_neg_values, base_config):
        """Analyze sensitivity to number of negative samples"""
        results = {}

        for num_neg in num_neg_values:
            print(f"Testing num_negative={num_neg}...")

            config = base_config.copy()
            config['num_negative'] = num_neg

            model = self.model_class(**config)
            self.train_func(model, config)
            metrics = self.eval_func(model)

            results[num_neg] = metrics

        return results

    def plot_sensitivity(self, results, param_name, metric_name='MRR', save_path=None):
        """Plot sensitivity analysis results"""
        param_values = sorted(results.keys())
        metric_values = [results[p][metric_name] for p in param_values]

        plt.figure(figsize=(10, 6))
        plt.plot(param_values, metric_values, marker='o', linewidth=2, markersize=8)
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'Sensitivity to {param_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


class GridSearchAnalyzer:
    """Grid search for hyperparameter tuning"""

    def __init__(self, model_class, train_func, eval_func):
        self.model_class = model_class
        self.train_func = train_func
        self.eval_func = eval_func

    def grid_search(self, param_grid, base_config):
        """
        Perform grid search over parameter combinations

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            base_config: Base configuration

        Returns:
            Results for all combinations
        """
        import itertools

        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        results = []

        for values in itertools.product(*param_values):
            config = base_config.copy()

            params = dict(zip(param_names, values))
            config.update(params)

            print(f"Testing {params}...")

            model = self.model_class(**config)
            self.train_func(model, config)
            metrics = self.eval_func(model)

            results.append({
                'params': params,
                'metrics': metrics
            })

        return results

    def get_best_params(self, results, metric_name='MRR'):
        """Get best parameter combination"""
        best_result = max(results, key=lambda x: x['metrics'][metric_name])
        return best_result['params'], best_result['metrics']


class ConvergenceAnalyzer:
    """Analyze training convergence"""

    def __init__(self):
        self.loss_history = []
        self.metric_history = []

    def record_loss(self, epoch, loss):
        """Record training loss"""
        self.loss_history.append((epoch, loss))

    def record_metrics(self, epoch, metrics):
        """Record evaluation metrics"""
        self.metric_history.append((epoch, metrics))

    def plot_convergence(self, save_path=None):
        """Plot convergence curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        epochs = [e for e, _ in self.loss_history]
        losses = [l for _, l in self.loss_history]

        axes[0].plot(epochs, losses, linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot MRR
        metric_epochs = [e for e, _ in self.metric_history]
        mrr_values = [m['MRR'] for _, m in self.metric_history]

        axes[1].plot(metric_epochs, mrr_values, linewidth=2, color='orange')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MRR', fontsize=12)
        axes[1].set_title('Validation MRR', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def detect_overfitting(self, threshold=0.05):
        """Detect overfitting based on metric trends"""
        if len(self.metric_history) < 10:
            return False

        recent_metrics = [m['MRR'] for _, m in self.metric_history[-10:]]
        max_mrr = max(recent_metrics)
        current_mrr = recent_metrics[-1]

        if max_mrr - current_mrr > threshold:
            return True

        return False
