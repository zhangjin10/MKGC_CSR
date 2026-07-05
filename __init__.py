"""
MKGC-CSR: Multimodal Knowledge Graph Completion with Causal Semantic Reasoning
"""

__version__ = '1.0.0'
__author__ = 'Binghong Li, Chuxin Xiao'
__email__ = '15384017902@163.com'

from models import MKGC_CSR
from data import create_data_loaders
from utils import setup_logger

__all__ = [
    'MKGC_CSR',
    'create_data_loaders',
    'setup_logger'
]
