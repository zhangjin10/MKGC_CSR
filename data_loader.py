"""
Data loading and preprocessing for MKGC-CSR
Handles FB15k-237-IMG and WN18-IMG datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms


class MultimodalKGDataset(Dataset):
    """Dataset for multimodal knowledge graph triplets"""

    def __init__(self, triplets, entity2id, relation2id,
                 text_features_dict, visual_features_dict,
                 num_entities, mode='train'):
        """
        Args:
            triplets: List of (head, relation, tail) triplets
            entity2id: Dictionary mapping entity names to IDs
            relation2id: Dictionary mapping relation names to IDs
            text_features_dict: Dictionary mapping entity IDs to text features
            visual_features_dict: Dictionary mapping entity IDs to visual features
            num_entities: Total number of entities
            mode: 'train', 'valid', or 'test'
        """
        self.triplets = triplets
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.text_features = text_features_dict
        self.visual_features = visual_features_dict
        self.num_entities = num_entities
        self.mode = mode

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        head, relation, tail = self.triplets[idx]

        # Get entity and relation IDs
        head_id = self.entity2id[head] if isinstance(head, str) else head
        relation_id = self.relation2id[relation] if isinstance(relation, str) else relation
        tail_id = self.entity2id[tail] if isinstance(tail, str) else tail

        # Get multimodal features
        head_text = self.text_features.get(head_id, torch.zeros(768))
        head_visual = self.visual_features.get(head_id, torch.zeros(768))
        tail_text = self.text_features.get(tail_id, torch.zeros(768))
        tail_visual = self.visual_features.get(tail_id, torch.zeros(768))

        return {
            'head': head_id,
            'relation': relation_id,
            'tail': tail_id,
            'head_text': head_text,
            'head_visual': head_visual,
            'tail_text': tail_text,
            'tail_visual': tail_visual
        }


class NegativeSampler:
    """Negative sampling for training"""

    def __init__(self, num_entities, num_negative=256):
        self.num_entities = num_entities
        self.num_negative = num_negative

    def sample(self, positive_triplets, corrupt_head=False):
        """
        Generate negative samples by corrupting head or tail entities

        Args:
            positive_triplets: Tensor of shape [batch_size, 3]
            corrupt_head: If True, corrupt head entities; otherwise corrupt tail
        Returns:
            negative_triplets: Tensor of shape [batch_size, num_negative, 3]
        """
        batch_size = positive_triplets.size(0)
        negative_triplets = positive_triplets.unsqueeze(1).repeat(1, self.num_negative, 1)

        # Random negative entities
        negative_entities = torch.randint(0, self.num_entities,
                                         (batch_size, self.num_negative))

        if corrupt_head:
            negative_triplets[:, :, 0] = negative_entities
        else:
            negative_triplets[:, :, 2] = negative_entities

        return negative_triplets


class DataPreprocessor:
    """Preprocess and load multimodal KG data"""

    def __init__(self, data_path, dataset_name='FB15k-237-IMG'):
        """
        Args:
            data_path: Path to dataset directory
            dataset_name: 'FB15k-237-IMG' or 'WN18-IMG'
        """
        self.data_path = data_path
        self.dataset_name = dataset_name

        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}

        self.train_triplets = []
        self.valid_triplets = []
        self.test_triplets = []

        self.text_features = {}
        self.visual_features = {}

    def load_triplets(self, file_path):
        """Load triplets from file"""
        triplets = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                head, relation, tail = line.strip().split('\t')
                triplets.append((head, relation, tail))
        return triplets

    def build_mappings(self):
        """Build entity and relation mappings"""
        entities = set()
        relations = set()

        # Collect all entities and relations
        for split in [self.train_triplets, self.valid_triplets, self.test_triplets]:
            for head, relation, tail in split:
                entities.add(head)
                entities.add(tail)
                relations.add(relation)

        # Create mappings
        self.entity2id = {entity: idx for idx, entity in enumerate(sorted(entities))}
        self.relation2id = {relation: idx for idx, relation in enumerate(sorted(relations))}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.id2relation = {idx: relation for relation, idx in self.relation2id.items()}

        print(f"Number of entities: {len(self.entity2id)}")
        print(f"Number of relations: {len(self.relation2id)}")

    def load_text_features(self, feature_path):
        """Load pre-extracted BERT text features"""
        if os.path.exists(feature_path):
            text_features = torch.load(feature_path)
            self.text_features = text_features
            print(f"Loaded text features from {feature_path}")
        else:
            print(f"Text features not found at {feature_path}, using random initialization")
            # Initialize random text features as placeholder
            for entity_id in range(len(self.entity2id)):
                self.text_features[entity_id] = torch.randn(768)

    def load_visual_features(self, feature_path):
        """Load pre-extracted ViT visual features"""
        if os.path.exists(feature_path):
            visual_features = torch.load(feature_path)
            self.visual_features = visual_features
            print(f"Loaded visual features from {feature_path}")
        else:
            print(f"Visual features not found at {feature_path}, using random initialization")
            # Initialize random visual features as placeholder
            for entity_id in range(len(self.entity2id)):
                self.visual_features[entity_id] = torch.randn(768)

    def prepare_datasets(self):
        """Prepare train, valid, and test datasets"""
        # Load triplets
        train_path = os.path.join(self.data_path, 'train.txt')
        valid_path = os.path.join(self.data_path, 'valid.txt')
        test_path = os.path.join(self.data_path, 'test.txt')

        if os.path.exists(train_path):
            self.train_triplets = self.load_triplets(train_path)
            print(f"Loaded {len(self.train_triplets)} training triplets")

        if os.path.exists(valid_path):
            self.valid_triplets = self.load_triplets(valid_path)
            print(f"Loaded {len(self.valid_triplets)} validation triplets")

        if os.path.exists(test_path):
            self.test_triplets = self.load_triplets(test_path)
            print(f"Loaded {len(self.test_triplets)} test triplets")

        # Build mappings
        self.build_mappings()

        # Load features
        text_feature_path = os.path.join(self.data_path, 'text_features.pt')
        visual_feature_path = os.path.join(self.data_path, 'visual_features.pt')

        self.load_text_features(text_feature_path)
        self.load_visual_features(visual_feature_path)

        # Convert triplets to ID format
        train_triplets_id = [(self.entity2id[h], self.relation2id[r], self.entity2id[t])
                             for h, r, t in self.train_triplets]
        valid_triplets_id = [(self.entity2id[h], self.relation2id[r], self.entity2id[t])
                             for h, r, t in self.valid_triplets]
        test_triplets_id = [(self.entity2id[h], self.relation2id[r], self.entity2id[t])
                            for h, r, t in self.test_triplets]

        # Create datasets
        train_dataset = MultimodalKGDataset(
            train_triplets_id, self.entity2id, self.relation2id,
            self.text_features, self.visual_features,
            len(self.entity2id), mode='train'
        )

        valid_dataset = MultimodalKGDataset(
            valid_triplets_id, self.entity2id, self.relation2id,
            self.text_features, self.visual_features,
            len(self.entity2id), mode='valid'
        )

        test_dataset = MultimodalKGDataset(
            test_triplets_id, self.entity2id, self.relation2id,
            self.text_features, self.visual_features,
            len(self.entity2id), mode='test'
        )

        return train_dataset, valid_dataset, test_dataset

    def get_all_triplets(self):
        """Get all triplets for filtered evaluation"""
        all_triplets = set()
        for h, r, t in self.train_triplets + self.valid_triplets + self.test_triplets:
            h_id = self.entity2id[h]
            r_id = self.relation2id[r]
            t_id = self.entity2id[t]
            all_triplets.add((h_id, r_id, t_id))
        return all_triplets


def create_data_loaders(data_path, dataset_name='FB15k-237-IMG',
                       batch_size=512, num_workers=4):
    """
    Create data loaders for training, validation, and testing

    Args:
        data_path: Path to dataset directory
        dataset_name: 'FB15k-237-IMG' or 'WN18-IMG'
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        train_loader, valid_loader, test_loader, preprocessor
    """
    preprocessor = DataPreprocessor(data_path, dataset_name)
    train_dataset, valid_dataset, test_dataset = preprocessor.prepare_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader, preprocessor


def collate_fn(batch):
    """Custom collate function for batching"""
    heads = torch.tensor([item['head'] for item in batch])
    relations = torch.tensor([item['relation'] for item in batch])
    tails = torch.tensor([item['tail'] for item in batch])

    head_texts = torch.stack([item['head_text'] for item in batch])
    head_visuals = torch.stack([item['head_visual'] for item in batch])
    tail_texts = torch.stack([item['tail_text'] for item in batch])
    tail_visuals = torch.stack([item['tail_visual'] for item in batch])

    return {
        'head': heads,
        'relation': relations,
        'tail': tails,
        'head_text': head_texts,
        'head_visual': head_visuals,
        'tail_text': tail_texts,
        'tail_visual': tail_visuals
    }
