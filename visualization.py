"""
Visualization utilities for MKGC-CSR
Visualize training curves, attention weights, and concept distributions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import os


def plot_training_curves(train_losses, valid_metrics, save_path=None):
    """
    Plot training loss and validation metrics over epochs

    Args:
        train_losses: List of training losses per epoch
        valid_metrics: List of dictionaries with validation metrics per epoch
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training loss
    axes[0].plot(train_losses, label='Training Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot validation metrics
    epochs = range(1, len(valid_metrics) + 1)
    mrr_scores = [m['MRR'] for m in valid_metrics]
    hits_at_1 = [m['Hits@1'] for m in valid_metrics]
    hits_at_3 = [m['Hits@3'] for m in valid_metrics]
    hits_at_10 = [m['Hits@10'] for m in valid_metrics]

    axes[1].plot(epochs, mrr_scores, label='MRR', linewidth=2, marker='o')
    axes[1].plot(epochs, hits_at_1, label='Hits@1', linewidth=2, marker='s')
    axes[1].plot(epochs, hits_at_3, label='Hits@3', linewidth=2, marker='^')
    axes[1].plot(epochs, hits_at_10, label='Hits@10', linewidth=2, marker='d')

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


def plot_robustness_comparison(results_dict, save_path=None):
    """
    Plot robustness comparison across different noise levels

    Args:
        results_dict: Dictionary mapping method names to robustness results
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for method_name, results in results_dict.items():
        noise_levels = sorted(results.keys())
        mrr_scores = [results[noise]['MRR'] for noise in noise_levels]
        hits_at_10 = [results[noise]['Hits@10'] for noise in noise_levels]

        axes[0].plot(noise_levels, mrr_scores, label=method_name,
                    linewidth=2.5, marker='o', markersize=8)
        axes[1].plot(noise_levels, hits_at_10, label=method_name,
                    linewidth=2.5, marker='o', markersize=8)

    axes[0].set_xlabel('Noise Standard Deviation (σ)', fontsize=12)
    axes[0].set_ylabel('MRR', fontsize=12)
    axes[0].set_title('MRR under Visual Noise', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Noise Standard Deviation (σ)', fontsize=12)
    axes[1].set_ylabel('Hits@10', fontsize=12)
    axes[1].set_title('Hits@10 under Visual Noise', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness comparison saved to {save_path}")
    else:
        plt.show()


def plot_visual_concepts(visual_dictionary, method='pca', save_path=None):
    """
    Visualize visual concept dictionary using dimensionality reduction

    Args:
        visual_dictionary: VisualConceptDictionary module
        method: 'pca' or 'tsne'
        save_path: Path to save the plot
    """
    centroids = visual_dictionary.get_centroids().detach().cpu().numpy()
    priors = visual_dictionary.get_priors().detach().cpu().numpy()

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(centroids)
        title = 'Visual Concepts (PCA)'
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42)
        coords = reducer.fit_transform(centroids)
        title = 'Visual Concepts (t-SNE)'

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1],
                         s=priors * 5000,  # Size proportional to prior
                         c=range(len(centroids)),
                         cmap='viridis',
                         alpha=0.6,
                         edgecolors='black',
                         linewidth=1)

    plt.colorbar(scatter, label='Concept Index')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visual concepts plot saved to {save_path}")
    else:
        plt.show()


def plot_concept_prior_distribution(visual_dictionary, save_path=None):
    """
    Plot distribution of visual concept priors

    Args:
        visual_dictionary: VisualConceptDictionary module
        save_path: Path to save the plot
    """
    priors = visual_dictionary.get_priors().detach().cpu().numpy()

    plt.figure(figsize=(12, 6))

    # Bar plot
    plt.subplot(1, 2, 1)
    plt.bar(range(len(priors)), priors, color='steelblue', edgecolor='black')
    plt.xlabel('Concept Index', fontsize=12)
    plt.ylabel('Prior Probability', fontsize=12)
    plt.title('Visual Concept Prior Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(priors, bins=20, color='coral', edgecolor='black', alpha=0.7)
    plt.xlabel('Prior Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prior Probability Histogram', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Concept prior distribution saved to {save_path}")
    else:
        plt.show()


def plot_relation_type_comparison(relation_type_results, save_path=None):
    """
    Plot performance comparison across different relation types

    Args:
        relation_type_results: Dictionary with results for each relation type
        save_path: Path to save the plot
    """
    relation_types = ['1-to-1', '1-to-N', 'N-to-1', 'N-to-N']
    head_mrr = []
    tail_mrr = []

    for rel_type in relation_types:
        if rel_type in relation_type_results:
            head_mrr.append(relation_type_results[rel_type]['head_prediction']['MRR'])
            tail_mrr.append(relation_type_results[rel_type]['tail_prediction']['MRR'])
        else:
            head_mrr.append(0)
            tail_mrr.append(0)

    x = np.arange(len(relation_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, head_mrr, width, label='Head Prediction',
                   color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, tail_mrr, width, label='Tail Prediction',
                   color='coral', edgecolor='black')

    ax.set_xlabel('Relation Type', fontsize=12)
    ax.set_ylabel('MRR', fontsize=12)
    ax.set_title('Performance by Relation Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(relation_types)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Relation type comparison saved to {save_path}")
    else:
        plt.show()


def plot_attention_heatmap(attention_weights, save_path=None):
    """
    Plot heatmap of causal attention weights over visual concepts

    Args:
        attention_weights: Attention weights [num_samples, num_concepts]
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    sns.heatmap(attention_weights[:50],  # Show first 50 samples
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=10,
                yticklabels=5)

    plt.xlabel('Visual Concept Index', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    plt.title('Causal Attention Weights Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved to {save_path}")
    else:
        plt.show()


def save_results_table(results, output_path):
    """
    Save results in a formatted table

    Args:
        results: Dictionary of results
        output_path: Path to save the table
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MKGC-CSR Results\n")
        f.write("="*80 + "\n\n")

        for key, value in results.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        f.write(f"  {sub_key}: {sub_value:.4f}\n")
                    else:
                        f.write(f"  {sub_key}: {sub_value}\n")
                f.write("\n")
            else:
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

        f.write("="*80 + "\n")

    print(f"Results table saved to {output_path}")
