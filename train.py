# Pokemon vs Digimon Binary Classifier with DinoV3 and Alignment Head

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import v2
from PIL import Image
import os
import glob
import random
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# DinoV3 Image Transform
def make_transform(resize_size: int = 1024):
    """Create DinoV3 compatible image transforms"""
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([resize, to_float, normalize])

# Change Detection Dataset Generator
def create_change_detection_pair_data_synth(root_dir='datasets/DV3SB-dataset-generator/dataset', image_size=1024):
    """
    Generate image pairs for binary classification from change detection dataset.
    Returns data in the same format as the original Pokemon vs Digimon function.
    
    Args:
        root_dir: Root directory containing 0/ (same pairs) and 1/ (different pairs)
        image_size: Target image size (default 224)
        
    Returns:
        Tuple of ((X_a_train, X_b_train), (X_a_val, X_b_val), (X_a_test, X_b_test), y_train, y_val, y_test)
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    root_path = Path(root_dir)
    pairs = []
    
    # Load all image pairs from the directory structure
    # 0/ contains same image pairs (label = 1)
    # 1/ contains different image pairs (label = 0)
    for label_dir in root_path.iterdir():
        if not label_dir.is_dir() or label_dir.name not in ['0', '1']:
            continue
        
        # Set label: 0 folder = same pairs (label=1), 1 folder = different pairs (label=0)
        label = 1 if label_dir.name == '0' else 0
        
        for pair_dir in label_dir.iterdir():
            if not pair_dir.is_dir():
                continue
            
            # Get all image files
            image_files = sorted([
                f for f in pair_dir.iterdir() 
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
            
            # Take first 2 images
            if len(image_files) >= 2:
                pairs.append({
                    'image1': image_files[0],
                    'image2': image_files[1],
                    'label': label
                })
    
    print(f"Found {len(pairs)} image pairs")
    
    if len(pairs) == 0:
        raise ValueError("No image pairs found! Check the dataset paths.")
    
    def load_and_preprocess_image(image_path, target_size=(1024, 1024)):
        """Load and preprocess an image"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize image
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Convert from HWC to CHW format
            img_array = np.transpose(img_array, (2, 0, 1))
            
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    images_a = []
    images_b = []
    labels = []
    
    # Process all pairs
    for pair in pairs:
        # Load and preprocess images
        img_a = load_and_preprocess_image(pair['image1'], (image_size, image_size))
        img_b = load_and_preprocess_image(pair['image2'], (image_size, image_size))
        
        # Skip if either image failed to load
        if img_a is None or img_b is None:
            continue
            
        images_a.append(img_a)
        images_b.append(img_b)
        labels.append(pair['label'])
    
    # Convert to numpy arrays
    X_a = np.array(images_a, dtype=np.float32)
    X_b = np.array(images_b, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    print(f"Successfully loaded {len(X_a)} image pairs")
    print(f"Same pairs: {np.sum(y)}")
    print(f"Different pairs: {len(y) - np.sum(y)}")
    
    # Split the data: 60% train, 20% validation, 20% test
    X_a_temp, X_a_test, X_b_temp, X_b_test, y_temp, y_test = train_test_split(
        X_a, X_b, y, test_size=0.2, random_state=42, stratify=y
    )
    X_a_train, X_a_val, X_b_train, X_b_val, y_train, y_val = train_test_split(
        X_a_temp, X_b_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # Print data split counts with same/different breakdown
    print(f"Set: same:different = total pairs")

    print(f"Train set: {np.sum(y_train)}:{len(y_train) - np.sum(y_train)} = {len(y_train)} pairs")
    print(f"Validation set: {np.sum(y_val)}:{len(y_val) - np.sum(y_val)} = {len(y_val)} pairs")
    print(f"Test set: {np.sum(y_test)}:{len(y_test) - np.sum(y_test)} = {len(y_test)} pairs")
    
    # Convert to PyTorch tensors and move to device
    X_a_train = torch.FloatTensor(X_a_train).to(device)
    X_a_val = torch.FloatTensor(X_a_val).to(device)
    X_a_test = torch.FloatTensor(X_a_test).to(device)
    X_b_train = torch.FloatTensor(X_b_train).to(device)
    X_b_val = torch.FloatTensor(X_b_val).to(device)
    X_b_test = torch.FloatTensor(X_b_test).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    y_test = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    return (X_a_train, X_b_train), (X_a_val, X_b_val), (X_a_test, X_b_test), y_train, y_val, y_test
    # return (X_a_train, X_b_train), y_train,

# Change Detection Dataset Generator
def create_change_detection_pair_data(root_dir='datasets/change-detection-natural-enviroments/train', image_size=1024):
    """
    Generate image pairs for binary classification from change detection dataset.
    Returns validation and test sets only.
    
    Args:
        root_dir: Root directory containing train/<source>/<label>/<pairs>
        image_size: Target image size (default 224)
        
    Returns:
        Tuple of ((X_a_val, X_b_val), (X_a_test, X_b_test), y_val, y_test)
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    root_path = Path(root_dir)
    pairs = []
    
    # Load all image pairs from the directory structure
    for source_dir in root_path.iterdir():
        if not source_dir.is_dir():
            continue
        
        for label_dir in source_dir.iterdir():
            if label_dir.name not in ['same', 'different']:
                continue
            
            label = 1 if label_dir.name == 'same' else 0
            
            for pair_dir in label_dir.iterdir():
                if not pair_dir.is_dir():
                    continue
                
                # Get all image files
                image_files = sorted([
                    f for f in pair_dir.iterdir() 
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
                ])
                
                # Take first 2 images
                if len(image_files) >= 2:
                    pairs.append({
                        'image1': image_files[0],
                        'image2': image_files[1],
                        'label': label
                    })
    
    print(f"Found {len(pairs)} image pairs")
    
    if len(pairs) == 0:
        raise ValueError("No image pairs found! Check the dataset paths.")
    
    def load_and_preprocess_image(image_path, target_size=(1024, 1024)):
        """Load and preprocess an image"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize image
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Convert from HWC to CHW format
            img_array = np.transpose(img_array, (2, 0, 1))
            
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    images_a = []
    images_b = []
    labels = []
    
    # Process all pairs
    for pair in pairs:
        # Load and preprocess images
        img_a = load_and_preprocess_image(pair['image1'], (image_size, image_size))
        img_b = load_and_preprocess_image(pair['image2'], (image_size, image_size))
        
        # Skip if either image failed to load
        if img_a is None or img_b is None:
            continue
            
        images_a.append(img_a)
        images_b.append(img_b)
        labels.append(pair['label'])
    
    # Convert to numpy arrays
    X_a = np.array(images_a, dtype=np.float32)
    X_b = np.array(images_b, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    print(f"Successfully loaded {len(X_a)} image pairs")
    print(f"Same pairs: {np.sum(y)}")
    print(f"Different pairs: {len(y) - np.sum(y)}")
    
    # Split the data: 50% validation, 50% test
    # X_a_val, X_a_test, X_b_val, X_b_test, y_val, y_test = train_test_split(
    #     X_a, X_b, y, test_size=0.5, random_state=42, stratify=y
    # )

    X_a_test = X_a
    X_b_test = X_b
    y_test = y

    # Print data split counts with same/different breakdown
    print(f"Set: same:different = total pairs")
    # print(f"Validation set: {np.sum(y_val)}:{len(y_val) - np.sum(y_val)} = {len(y_val)} pairs")
    print(f"Test set: {np.sum(y_test)}:{len(y_test) - np.sum(y_test)} = {len(y_test)} pairs")
    
    # Convert to PyTorch tensors and move to device
    # X_a_val = torch.FloatTensor(X_a_val).to(device)
    X_a_test = torch.FloatTensor(X_a_test).to(device)
    # X_b_val = torch.FloatTensor(X_b_val).to(device)
    X_b_test = torch.FloatTensor(X_b_test).to(device)
    # y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    y_test = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    # return (X_a_val, X_b_val), (X_a_test, X_b_test), y_val, y_test
    return (X_a_test, X_b_test), y_test

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        """
        Contrastive loss function.
        Args:
            output1: First embedding [batch_size, embedding_dim]
            output2: Second embedding [batch_size, embedding_dim]
            label: Binary labels [batch_size, 1] - 1 for same, 0 for different
        """
        # Calculate euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Contrastive loss calculation
        # For same pairs (label=1): minimize distance
        # For different pairs (label=0): maximize distance up to margin
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

# DinoV3 Siamese Network with Contrastive Loss
class DinoV3ContrastiveClassifier(nn.Module):
    def __init__(self, repo_dir, model_name='dinov3_vits16', weights_path=None, 
                 embedding_dim=1024, freeze_backbone=True, margin=1.0):
        super(DinoV3ContrastiveClassifier, self).__init__()
        
        # Load DinoV3 backbone from local directory
        if weights_path:
            self.backbone = torch.hub.load(repo_dir, model_name, source='local', weights=weights_path)
        else:
            self.backbone = torch.hub.load(repo_dir, model_name, source='local')
        
        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the feature dimension from DinoV3
        feature_dims = {
            'dinov3_vits16': 384,
            'dinov3_vits16plus': 384,
            'dinov3_vitb16': 768,
            'dinov3_vitl16': 1024,
            'dinov3_vith16plus': 1280,
            'dinov3_vit7b16': 1536,
            'dinov3_convnext_tiny': 768,
            'dinov3_convnext_small': 768,
            'dinov3_convnext_base': 1024,
            'dinov3_convnext_large': 1536,
        }
        
        self.feature_dim = feature_dims.get(model_name, 384)
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        self.margin = margin
        
        # Image preprocessing using DinoV3 official transforms
        self.transform = make_transform(resize_size=1024)
        
        # Shared embedding head (projects features to embedding space)
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)  # L2 normalize embeddings
        )
        
        # Contrastive loss function
        self.contrastive_loss = ContrastiveLoss(margin=margin)
        
        # Move model to device
        self.to(device)
        
    def extract_features(self, x):
        """Extract features from a single image using DinoV3"""
        # Apply preprocessing transforms
        x = self.transform(x)
        
        # Extract features using DinoV3 backbone
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)  # [batch_size, feature_dim]
        else:
            features = self.backbone(x)  # [batch_size, feature_dim]
        
        return features
    
    def get_embeddings(self, x):
        """Get normalized embeddings for an image"""
        features = self.extract_features(x)
        embeddings = self.embedding_head(features)
        # L2 normalize embeddings for better contrastive learning
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
        
    def forward(self, x_a, x_b):
        """
        Forward pass for image pair comparison
        Args:
            x_a: First image tensor [batch_size, 3, H, W]
            x_b: Second image tensor [batch_size, 3, H, W]
        Returns:
            Tuple of (embedding_a, embedding_b) for contrastive loss
        """
        # Get normalized embeddings from both images
        embedding_a = self.get_embeddings(x_a)  # [batch_size, embedding_dim]
        embedding_b = self.get_embeddings(x_b)  # [batch_size, embedding_dim]
        
        return embedding_a, embedding_b
    
    def predict_similarity(self, x_a, x_b, threshold=0.5):
        """
        Predict similarity between two images based on distance threshold
        Args:
            x_a: First image tensor [batch_size, 3, H, W]
            x_b: Second image tensor [batch_size, 3, H, W]
            threshold: Distance threshold for similarity (lower = more similar)
        Returns:
            Binary predictions [batch_size, 1] - 1 for same, 0 for different
        """
        embedding_a, embedding_b = self.forward(x_a, x_b)
        distances = F.pairwise_distance(embedding_a, embedding_b, keepdim=True)
        predictions = (distances < threshold).float()
        return predictions, distances

# Training Function with Contrastive Loss and Mini-batching
def train_model_contrastive(model, X_train_pair, y_train, X_val_pair, y_val, epochs=100, lr=0.001, batch_size=32, distance_threshold=0.5):
    """Train the contrastive model with mini-batching"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    X_a_train, X_b_train = X_train_pair
    X_a_val, X_b_val = X_val_pair
    
    n_train_samples = X_a_train.shape[0]
    n_val_samples = X_a_val.shape[0]
    
    for epoch in range(epochs):
        # Training with mini-batches
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0
        
        # Shuffle training data
        indices = torch.randperm(n_train_samples)
        X_a_train_shuffled = X_a_train[indices]
        X_b_train_shuffled = X_b_train[indices]
        y_train_shuffled = y_train[indices]
        
        for i in range(0, n_train_samples, batch_size):
            end_idx = min(i + batch_size, n_train_samples)
            
            # Get mini-batch
            batch_x_a = X_a_train_shuffled[i:end_idx]
            batch_x_b = X_b_train_shuffled[i:end_idx]
            batch_y = y_train_shuffled[i:end_idx]
            
            optimizer.zero_grad()
            
            # Forward pass - get embeddings
            embedding_a, embedding_b = model(batch_x_a, batch_x_b)
            
            # Calculate contrastive loss
            loss = model.contrastive_loss(embedding_a, embedding_b, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            n_train_batches += 1
        
        avg_train_loss = epoch_train_loss / n_train_batches
        train_losses.append(avg_train_loss)
        
        # Validation with mini-batches
        model.eval()
        epoch_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for i in range(0, n_val_samples, batch_size):
                end_idx = min(i + batch_size, n_val_samples)
                
                # Get mini-batch
                batch_x_a = X_a_val[i:end_idx]
                batch_x_b = X_b_val[i:end_idx]
                batch_y = y_val[i:end_idx]
                
                # Forward pass - get embeddings
                embedding_a, embedding_b = model(batch_x_a, batch_x_b)
                
                # Calculate contrastive loss
                val_loss = model.contrastive_loss(embedding_a, embedding_b, batch_y)
                
                # Calculate predictions based on distance threshold
                distances = F.pairwise_distance(embedding_a, embedding_b, keepdim=True)
                val_predictions = (distances < distance_threshold).float()
                
                # Accumulate loss and accuracy
                epoch_val_loss += val_loss.item()
                correct_predictions += (val_predictions == batch_y).float().sum().item()
                total_predictions += batch_y.size(0)
                n_val_batches += 1
        
        avg_val_loss = epoch_val_loss / n_val_batches
        val_accuracy = correct_predictions / total_predictions
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Print progress
        if epoch % 1 == 0:
            print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    return train_losses, val_losses, val_accuracies

# Inference Function with Contrastive Model and Threshold Optimization
def run_inference_contrastive(model, X_test_pair, y_test, batch_size=32, optimize_threshold=True):
    """Run inference and evaluate the contrastive model with mini-batching"""
    model.eval()
    X_a_test, X_b_test = X_test_pair
    n_test_samples = X_a_test.shape[0]
    
    all_distances = []
    
    # First pass: collect all distances
    with torch.no_grad():
        for i in range(0, n_test_samples, batch_size):
            end_idx = min(i + batch_size, n_test_samples)
            
            # Get mini-batch
            batch_x_a = X_a_test[i:end_idx]
            batch_x_b = X_b_test[i:end_idx]
            
            # Forward pass - get embeddings
            embedding_a, embedding_b = model(batch_x_a, batch_x_b)
            
            # Calculate distances
            distances = F.pairwise_distance(embedding_a, embedding_b, keepdim=True)
            all_distances.append(distances)
        
        # Concatenate all distances
        all_distances = torch.cat(all_distances, dim=0)
    
    # Optimize threshold if requested
    best_threshold = 0.5
    best_accuracy = 0.0
    
    if optimize_threshold:
        print("Optimizing distance threshold...")
        thresholds = torch.linspace(0.1, 2.0, 100)
        
        for threshold in thresholds:
            predictions = (all_distances < threshold).float()
            accuracy = (predictions == y_test).float().mean().item()
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold.item()
        
        print(f"Best threshold: {best_threshold:.4f} with accuracy: {best_accuracy:.4f}")
    
    # Final predictions with best threshold
    all_predictions = (all_distances < best_threshold).float()
    
    # Calculate metrics
    accuracy = (all_predictions == y_test).float().mean()
    
    # Calculate precision, recall, F1
    tp = ((all_predictions == 1) & (y_test == 1)).float().sum()
    fp = ((all_predictions == 1) & (y_test == 0)).float().sum()
    fn = ((all_predictions == 0) & (y_test == 1)).float().sum()
    tn = ((all_predictions == 0) & (y_test == 0)).float().sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-class accuracy
    class_0_total = (y_test == 0).float().sum()  # Different pairs
    class_1_total = (y_test == 1).float().sum()  # Same pairs
    
    class_0_correct = tn  # Correctly predicted as different
    class_1_correct = tp  # Correctly predicted as same
    
    class_0_accuracy = class_0_correct / class_0_total if class_0_total > 0 else 0
    class_1_accuracy = class_1_correct / class_1_total if class_1_total > 0 else 0
    
    # Distance statistics
    same_distances = all_distances[y_test == 1]
    diff_distances = all_distances[y_test == 0]
    
    print(f"\nDistance Statistics:")
    print(f"Same pairs - Mean: {same_distances.mean():.4f}, Std: {same_distances.std():.4f}")
    print(f"Different pairs - Mean: {diff_distances.mean():.4f}, Std: {diff_distances.std():.4f}")
    print(f"Optimal threshold: {best_threshold:.4f}")
    
    print(f"\nFinal Results:")
    print(f"Overall Accuracy: {accuracy.item():.4f}")
    print(f"Precision: {precision.item():.4f}")
    print(f"Recall: {recall.item():.4f}")
    print(f"F1 Score: {f1.item():.4f}")
    print(f"\nPer-Class Accuracy:")
    print(f"Class 0 (Different): {class_0_accuracy.item():.4f} ({class_0_correct.int()}/{class_0_total.int()})")
    print(f"Class 1 (Same): {class_1_accuracy.item():.4f} ({class_1_correct.int()}/{class_1_total.int()})")
    
    return accuracy.item(), precision.item(), recall.item(), f1.item(), best_threshold

def main():
    """Main function to run change detection contrastive classifier"""
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Running on CPU.")
    
    print("\n" + "=" * 60)
    print("CHANGE DETECTION CONTRASTIVE CLASSIFIER")
    print("DINOV3 BACKBONE WITH CONTRASTIVE LOSS")
    print("=" * 60)
    
    # Configuration for local DinoV3 loading
    repo_dir = "./dino/dinov3-main"
    model_name = "dinov3_vitl16"
    weights_path = "./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    
    # Check if repo directory exists
    if not os.path.exists(repo_dir):
        print(f"Error: DinoV3 repository directory '{repo_dir}' not found!")
        print("Please clone the DinoV3 repository first:")
        print("git clone https://github.com/facebookresearch/dinov3.git")
        return
    
    # Create change detection dataset
    print("Creating change detection image pair data for binary classification...")
    X_train_pair, X_val_pair, X_test_pair, y_train, y_val, y_test = create_change_detection_pair_data_synth(
    # X_train_pair, y_train = create_change_detection_pair_data_synth(
        root_dir='datasets/DV3SB-dataset-generator/dataset', 
        image_size=1024
    )

    # X_test_pair, y_test = create_change_detection_pair_data(
    #     root_dir='datasets/change-detection-natural-enviroments/train', 
    #     image_size=224
    # )
    
    X_a_train, X_b_train = X_train_pair
    X_a_val, X_b_val = X_val_pair
    X_a_test, X_b_test = X_test_pair
    
    print(f"Training data shapes: Image A: {X_a_train.shape}, Image B: {X_b_train.shape}")
    print(f"Validation data shapes: Image A: {X_a_val.shape}, Image B: {X_b_val.shape}")
    print(f"Test data shapes: Image A: {X_a_test.shape}, Image B: {X_b_test.shape}")
    print(f"Labels - Train: {y_train.shape}, Val: {y_val.shape}, Test: {y_test.shape}")
    
    # Initialize contrastive model
    print(f"Loading DinoV3 contrastive classifier from: {repo_dir}")
    try:
        model = DinoV3ContrastiveClassifier(
            repo_dir=repo_dir,
            model_name=model_name,
            weights_path=weights_path,
            embedding_dim=2048,
            freeze_backbone=True,
            margin=1.0
        )
        
        print(f"\nModel architecture:")
        print(f"Backbone: DinoV3 {model_name}")
        print(f"Feature dimension: {model.feature_dim}")
        print(f"Embedding dimension: {model.embedding_dim}")
        print(f"Contrastive margin: {model.margin}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Backbone frozen: {model.freeze_backbone}")
        
    except Exception as e:
        print(f"Error loading DinoV3 model: {e}")
        print("Make sure you have:")
        print("1. Cloned the DinoV3 repository")
        print("2. Downloaded the appropriate weights")
        print("3. Set the correct repo_dir and weights_path")
        return
    
    # Train the model with contrastive loss and mini-batching
    print(f"\nStarting contrastive training with mini-batching...")
    batch_size = 4  # Adjust based on your RAM capacity
    distance_threshold = 0.5  # Initial threshold for validation accuracy calculation
    
    train_losses, val_losses, val_accuracies = train_model_contrastive(
        model, X_train_pair, y_train, X_val_pair, y_val, 
        epochs=30, lr=0.0001, batch_size=batch_size, distance_threshold=distance_threshold
    )
    
    # Run final inference on test set with threshold optimization
    accuracy, precision, recall, f1, optimal_threshold = run_inference_contrastive(
        model, X_test_pair, y_test, batch_size=batch_size, optimize_threshold=True
    )
    
    # Save the trained contrastive model
    model_save_path = "change_detection_contrastive_classifier.pth"
    print(f"\nSaving trained contrastive model to: {model_save_path}")
    
    # Save model state dict and training info
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'repo_dir': repo_dir,
            'model_name': model_name,
            'weights_path': weights_path,
            'embedding_dim': model.embedding_dim,
            'feature_dim': model.feature_dim,
            'freeze_backbone': model.freeze_backbone,
            'margin': model.margin
        },
        'training_config': {
            'epochs': 50,
            'batch_size': batch_size,
            'learning_rate': 0.0001,
            'distance_threshold': distance_threshold,
            'loss_type': 'contrastive',
            'dataset': 'change-detection-natural-enviroments'
        },
        'final_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'optimal_threshold': optimal_threshold
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    }, model_save_path)
    
    print(f"Model saved successfully!")
    print(f"To load the model later, use:")
    print(f"checkpoint = torch.load('{model_save_path}')")
    print(f"model.load_state_dict(checkpoint['model_state_dict'])")
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()