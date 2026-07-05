"""
Utility functions for feature extraction
Extract text features using BERT and visual features using ViT
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel, ViTFeatureExtractor
from PIL import Image
import os
from tqdm import tqdm
import json


class TextFeatureExtractor:
    """Extract text features using BERT"""

    def __init__(self, model_name='bert-base-uncased', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_features(self, texts, batch_size=32):
        """
        Extract BERT features for a list of texts

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping indices to feature vectors
        """
        features = {}

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
                batch_texts = texts[i:i+batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)

                # Extract features
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]

                # Store features
                for j, embedding in enumerate(cls_embeddings):
                    features[i + j] = embedding.cpu()

        return features

    def extract_from_entity_descriptions(self, entity_descriptions):
        """
        Extract features from entity descriptions

        Args:
            entity_descriptions: Dictionary mapping entity IDs to text descriptions

        Returns:
            Dictionary mapping entity IDs to feature tensors
        """
        entity_ids = list(entity_descriptions.keys())
        texts = [entity_descriptions[eid] for eid in entity_ids]

        features_list = self.extract_features(texts)

        # Map back to entity IDs
        features = {entity_ids[i]: feat for i, feat in features_list.items()}

        return features


class VisualFeatureExtractor:
    """Extract visual features using Vision Transformer (ViT)"""

    def __init__(self, model_name='google/vit-base-patch16-224', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_single_image(self, image_path):
        """
        Extract features from a single image

        Args:
            image_path: Path to image file

        Returns:
            Feature tensor of shape [768]
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.feature_extractor(images=image, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                features = outputs.last_hidden_state[:, 0, :].squeeze()

            return features.cpu()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return torch.zeros(768)

    def extract_multiple_images(self, image_paths, aggregate='mean'):
        """
        Extract features from multiple images and aggregate

        Args:
            image_paths: List of image paths
            aggregate: Aggregation method ('mean', 'max', 'first')

        Returns:
            Aggregated feature tensor of shape [768]
        """
        features_list = []

        for img_path in image_paths:
            feat = self.extract_single_image(img_path)
            features_list.append(feat)

        if len(features_list) == 0:
            return torch.zeros(768)

        features_tensor = torch.stack(features_list)

        if aggregate == 'mean':
            return features_tensor.mean(dim=0)
        elif aggregate == 'max':
            return features_tensor.max(dim=0)[0]
        elif aggregate == 'first':
            return features_tensor[0]
        else:
            return features_tensor.mean(dim=0)

    def extract_from_entity_images(self, entity_image_paths, aggregate='mean'):
        """
        Extract visual features for all entities

        Args:
            entity_image_paths: Dictionary mapping entity IDs to lists of image paths
            aggregate: Aggregation method for multiple images per entity

        Returns:
            Dictionary mapping entity IDs to feature tensors
        """
        features = {}

        for entity_id, image_paths in tqdm(entity_image_paths.items(),
                                          desc="Extracting visual features"):
            if len(image_paths) > 0:
                features[entity_id] = self.extract_multiple_images(image_paths, aggregate)
            else:
                features[entity_id] = torch.zeros(768)

        return features


def prepare_multimodal_features(data_path, dataset_name='FB15k-237-IMG',
                                output_path=None, device='cuda'):
    """
    Prepare text and visual features for a dataset

    Args:
        data_path: Path to dataset directory
        dataset_name: Dataset name
        output_path: Path to save extracted features
        device: Device for feature extraction

    Returns:
        text_features, visual_features dictionaries
    """
    if output_path is None:
        output_path = data_path

    print(f"Preparing multimodal features for {dataset_name}...")

    # Load entity information
    entity_desc_path = os.path.join(data_path, 'entity_descriptions.json')
    entity_images_path = os.path.join(data_path, 'entity_images.json')

    # Text feature extraction
    text_extractor = TextFeatureExtractor(device=device)

    if os.path.exists(entity_desc_path):
        print("Loading entity descriptions...")
        with open(entity_desc_path, 'r') as f:
            entity_descriptions = json.load(f)

        print("Extracting text features...")
        text_features = text_extractor.extract_from_entity_descriptions(entity_descriptions)

        # Save text features
        text_feature_path = os.path.join(output_path, 'text_features.pt')
        torch.save(text_features, text_feature_path)
        print(f"Saved text features to {text_feature_path}")
    else:
        print("Entity descriptions not found, skipping text extraction")
        text_features = {}

    # Visual feature extraction
    visual_extractor = VisualFeatureExtractor(device=device)

    if os.path.exists(entity_images_path):
        print("Loading entity image paths...")
        with open(entity_images_path, 'r') as f:
            entity_images = json.load(f)

        print("Extracting visual features...")
        visual_features = visual_extractor.extract_from_entity_images(entity_images)

        # Save visual features
        visual_feature_path = os.path.join(output_path, 'visual_features.pt')
        torch.save(visual_features, visual_feature_path)
        print(f"Saved visual features to {visual_feature_path}")
    else:
        print("Entity images not found, skipping visual extraction")
        visual_features = {}

    return text_features, visual_features


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract multimodal features')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default='FB15k-237-IMG',
                       help='Dataset name')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save features')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for extraction')

    args = parser.parse_args()

    prepare_multimodal_features(
        args.data_path,
        args.dataset_name,
        args.output_path,
        args.device
    )
