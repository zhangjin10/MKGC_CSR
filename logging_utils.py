"""
Logging utilities
"""

import logging
import os
import sys
from datetime import datetime
import json


def setup_logger(name, log_dir='./logs', level=logging.INFO):
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level

    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_metrics(logger, metrics, prefix='', epoch=None):
    """
    Log metrics to logger

    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        prefix: Prefix for log message
        epoch: Optional epoch number
    """
    if epoch is not None:
        log_msg = f"{prefix} Epoch {epoch}: "
    else:
        log_msg = f"{prefix} "

    metric_strs = [f"{k}={v:.4f}" for k, v in metrics.items()]
    log_msg += ", ".join(metric_strs)

    logger.info(log_msg)


class MetricsLogger:
    """Logger for tracking metrics over training"""

    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.metrics_history = {
            'train': [],
            'valid': [],
            'test': []
        }

    def log(self, split, epoch, metrics):
        """Log metrics for a split at an epoch"""
        entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history[split].append(entry)

    def save(self, filename='metrics_history.json'):
        """Save metrics history to file"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)

    def load(self, filename='metrics_history.json'):
        """Load metrics history from file"""
        filepath = os.path.join(self.log_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.metrics_history = json.load(f)

    def get_best_epoch(self, split='valid', metric='MRR'):
        """Get epoch with best metric value"""
        if not self.metrics_history[split]:
            return None

        best_epoch = max(
            self.metrics_history[split],
            key=lambda x: x.get(metric, 0)
        )

        return best_epoch['epoch']


class TensorboardLogger:
    """Logger for Tensorboard"""

    def __init__(self, log_dir='./runs'):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            self.enabled = False

    def log_scalar(self, tag, value, step):
        """Log scalar value"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars"""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag, values, step):
        """Log histogram"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def close(self):
        """Close writer"""
        if self.enabled:
            self.writer.close()
