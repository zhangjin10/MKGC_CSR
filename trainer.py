"""
Training script for MKGC-CSR model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.cluster import KMeans


class MKGC_CSR_Trainer:
    """Trainer for MKGC-CSR model"""

    def __init__(self, model, train_loader, valid_loader, test_loader,
                 preprocessor, config):
        """
        Args:
            model: MKGC_CSR model
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            preprocessor: Data preprocessor
            config: Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.preprocessor = preprocessor
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )

        # Loss function
        self.criterion = nn.MarginRankingLoss(margin=config.get('margin', 1.0))

        # Training parameters
        self.num_epochs = config.get('num_epochs', 500)
        self.num_negative = config.get('num_negative', 256)
        self.patience = config.get('patience', 50)
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Best validation metric
        self.best_valid_mrr = 0.0
        self.patience_counter = 0

    def initialize_visual_dictionary(self):
        """
        Initialize visual concept dictionary using K-Means clustering
        on training visual features
        """
        print("Initializing visual concept dictionary with K-Means...")

        # Collect all visual features from training data
        all_visual_features = []
        entity_ids = []

        for entity_id in range(len(self.preprocessor.entity2id)):
            if entity_id in self.preprocessor.visual_features:
                visual_feat = self.preprocessor.visual_features[entity_id]
                all_visual_features.append(visual_feat.cpu().numpy())
                entity_ids.append(entity_id)

        all_visual_features = np.array(all_visual_features)
        print(f"Clustering {len(all_visual_features)} visual features...")

        # Perform K-Means clustering
        num_concepts = self.model.visual_dictionary.num_concepts
        kmeans = KMeans(n_clusters=num_concepts, random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_visual_features)

        # Count cluster assignments
        counts = np.bincount(labels, minlength=num_concepts)

        # Update visual dictionary
        centroids_tensor = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        counts_tensor = torch.tensor(counts, dtype=torch.float32)

        visual_features_tensor = torch.tensor(all_visual_features, dtype=torch.float32)

        self.model.visual_dictionary.update_from_kmeans(
            visual_features_tensor,
            labels_tensor,
            counts_tensor
        )

        print(f"Visual dictionary initialized with {num_concepts} concepts")
        print(f"Cluster distribution: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    def self_adversarial_negative_sampling(self, positive_scores, negative_scores, temperature=1.0):
        """
        Self-adversarial negative sampling weighting
        """
        # Softmax over negative scores
        weights = torch.softmax(negative_scores * temperature, dim=-1).detach()
        return weights

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            heads = batch['head'].to(self.device)
            relations = batch['relation'].to(self.device)
            tails = batch['tail'].to(self.device)
            head_texts = batch['head_text'].to(self.device)
            head_visuals = batch['head_visual'].to(self.device)
            tail_texts = batch['tail_text'].to(self.device)
            tail_visuals = batch['tail_visual'].to(self.device)

            batch_size = heads.size(0)

            # Positive scores
            positive_scores = self.model(
                heads, relations, tails,
                head_texts, head_visuals,
                tail_texts, tail_visuals
            )

            # Negative sampling - corrupt tail
            negative_tails = torch.randint(
                0, self.model.num_entities,
                (batch_size, self.num_negative),
                device=self.device
            )

            # Compute negative scores
            negative_scores_list = []
            for i in range(self.num_negative):
                neg_tails = negative_tails[:, i]
                # Get negative tail features
                neg_tail_texts = torch.stack([
                    self.preprocessor.text_features.get(tid.item(), torch.zeros(768))
                    for tid in neg_tails
                ]).to(self.device)

                neg_tail_visuals = torch.stack([
                    self.preprocessor.visual_features.get(tid.item(), torch.zeros(768))
                    for tid in neg_tails
                ]).to(self.device)

                neg_scores = self.model(
                    heads, relations, neg_tails,
                    head_texts, head_visuals,
                    neg_tail_texts, neg_tail_visuals
                )
                negative_scores_list.append(neg_scores)

            negative_scores = torch.stack(negative_scores_list, dim=1)  # [batch_size, num_negative]

            # Self-adversarial weighting
            weights = self.self_adversarial_negative_sampling(positive_scores, negative_scores)

            # Weighted negative scores
            weighted_negative_scores = (negative_scores * weights).sum(dim=1)

            # Margin ranking loss
            target = torch.ones_like(positive_scores)
            loss = self.criterion(positive_scores, weighted_negative_scores, target)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(self, data_loader, mode='valid'):
        """
        Evaluate model on validation or test set
        Returns MRR, Hits@1, Hits@3, Hits@10
        """
        self.model.eval()

        all_ranks = []
        all_triplets = self.preprocessor.get_all_triplets()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {mode}"):
                heads = batch['head'].to(self.device)
                relations = batch['relation'].to(self.device)
                tails = batch['tail'].to(self.device)
                head_texts = batch['head_text'].to(self.device)
                head_visuals = batch['head_visual'].to(self.device)

                batch_size = heads.size(0)

                # Get scores for all possible tails
                for i in range(batch_size):
                    h = heads[i:i+1]
                    r = relations[i:i+1]
                    t_true = tails[i].item()
                    h_text = head_texts[i:i+1]
                    h_visual = head_visuals[i:i+1]

                    # Score all entities as potential tails
                    scores = []
                    for t_candidate in range(self.model.num_entities):
                        t_tensor = torch.tensor([t_candidate], device=self.device)

                        t_text = self.preprocessor.text_features.get(
                            t_candidate, torch.zeros(768)
                        ).unsqueeze(0).to(self.device)

                        t_visual = self.preprocessor.visual_features.get(
                            t_candidate, torch.zeros(768)
                        ).unsqueeze(0).to(self.device)

                        score = self.model(
                            h, r, t_tensor,
                            h_text, h_visual,
                            t_text, t_visual
                        )
                        scores.append(score.item())

                    scores = np.array(scores)

                    # Filtered ranking: remove other true triplets
                    h_item = h.item()
                    r_item = r.item()
                    for t_other in range(self.model.num_entities):
                        if t_other != t_true and (h_item, r_item, t_other) in all_triplets:
                            scores[t_other] = -1e10  # Very low score

                    # Get rank of true tail
                    sorted_indices = np.argsort(-scores)  # Descending order
                    rank = np.where(sorted_indices == t_true)[0][0] + 1
                    all_ranks.append(rank)

        # Compute metrics
        all_ranks = np.array(all_ranks)
        mrr = np.mean(1.0 / all_ranks)
        hits_at_1 = np.mean(all_ranks <= 1)
        hits_at_3 = np.mean(all_ranks <= 3)
        hits_at_10 = np.mean(all_ranks <= 10)

        metrics = {
            'MRR': mrr,
            'Hits@1': hits_at_1,
            'Hits@3': hits_at_3,
            'Hits@10': hits_at_10
        }

        return metrics

    def train(self):
        """Main training loop"""
        print("Starting training...")

        # Initialize visual dictionary
        self.initialize_visual_dictionary()

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")

            # Validate every few epochs
            if (epoch + 1) % 5 == 0:
                valid_metrics = self.evaluate(self.valid_loader, mode='valid')
                print(f"Validation Metrics: MRR={valid_metrics['MRR']:.4f}, "
                      f"Hits@1={valid_metrics['Hits@1']:.4f}, "
                      f"Hits@3={valid_metrics['Hits@3']:.4f}, "
                      f"Hits@10={valid_metrics['Hits@10']:.4f}")

                # Early stopping check
                if valid_metrics['MRR'] > self.best_valid_mrr:
                    self.best_valid_mrr = valid_metrics['MRR']
                    self.patience_counter = 0

                    # Save best model
                    self.save_checkpoint(epoch, valid_metrics, is_best=True)
                    print("Saved best model!")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        print("\nTraining completed!")
        print(f"Best validation MRR: {self.best_valid_mrr:.4f}")

        # Load best model and evaluate on test set
        self.load_checkpoint(os.path.join(self.checkpoint_dir, 'best_model.pt'))
        test_metrics = self.evaluate(self.test_loader, mode='test')
        print(f"\nTest Metrics: MRR={test_metrics['MRR']:.4f}, "
              f"Hits@1={test_metrics['Hits@1']:.4f}, "
              f"Hits@3={test_metrics['Hits@3']:.4f}, "
              f"Hits@10={test_metrics['Hits@10']:.4f}")

        return test_metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filepath}")
