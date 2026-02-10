# Gradio Frontend for Pokemon vs Digimon Binary Classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import v2
from PIL import Image
import gradio as gr
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# DinoV3 Image Transform
def make_transform(resize_size: int = 224):
    """Create DinoV3 compatible image transforms"""
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([resize, to_float, normalize])

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

# DinoV3 Siamese Network with Contrastive Loss (same as training)
class DinoV3ContrastiveClassifier(nn.Module):
    def __init__(self, repo_dir, model_name='dinov3_vits16', weights_path=None, 
                 embedding_dim=256, freeze_backbone=True, margin=1.0):
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
        self.transform = make_transform(resize_size=224)
        
        # Shared embedding head (projects features to embedding space)
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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

def load_trained_model(model_path="change_detection_contrastive_classifier.pth"):
    """Load the trained model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Initialize model with saved configuration
    model = DinoV3ContrastiveClassifier(
        repo_dir=model_config['repo_dir'],
        model_name=model_config['model_name'],
        weights_path=model_config['weights_path'],
        embedding_dim=model_config['embedding_dim'],
        freeze_backbone=model_config['freeze_backbone'],
        margin=model_config['margin']
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Model configuration: {model_config}")
    print(f"Final training metrics: {checkpoint['final_metrics']}")
    
    return model, checkpoint

def preprocess_image(image):
    """Preprocess PIL image for model input"""
    if image is None:
        return None
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Convert from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.FloatTensor(img_array).unsqueeze(0).to(device)
    
    return img_tensor

def predict_similarity(image1, image2, model, checkpoint):
    """Predict similarity between two images using contrastive model"""
    if image1 is None or image2 is None:
        return "Please upload both images", 0.0, "N/A"
    
    try:
        # Preprocess images
        img1_tensor = preprocess_image(image1)
        img2_tensor = preprocess_image(image2)
        
        if img1_tensor is None or img2_tensor is None:
            return "Error processing images", 0.0, "N/A"
        
        # Run inference using the contrastive model
        with torch.no_grad():
            # Get embeddings from both images
            embedding_a, embedding_b = model(img1_tensor, img2_tensor)
            
            # Calculate distance between embeddings
            distance = F.pairwise_distance(embedding_a, embedding_b, keepdim=True).item()
            
            # Use optimal threshold from training if available
            optimal_threshold = checkpoint['final_metrics'].get('optimal_threshold', 0.5)
            
            # Predict based on distance threshold
            is_same = distance < optimal_threshold
            
        # Interpret results
        if is_same:
            prediction = "Same"
            # Convert distance to similarity score (lower distance = higher similarity)
            similarity_score = max(0.0, 1.0 - (distance / optimal_threshold))
        else:
            prediction = "Different"
            # Convert distance to dissimilarity score
            similarity_score = min(1.0, distance / optimal_threshold) - 1.0
            similarity_score = abs(similarity_score)  # Make it positive
        
        result_text = f"Prediction: {prediction}\nDistance: {distance:.4f}\nThreshold: {optimal_threshold:.4f}\nSimilarity Score: {similarity_score:.3f}"
        
        return result_text, similarity_score, prediction
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(error_msg)
        return error_msg, 0.0, "Error"

def create_gradio_interface():
    """Create Gradio interface for the model"""
    
    # Load the trained model
    try:
        model, checkpoint = load_trained_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'change_detection_contrastive_classifier.pth' exists in the current directory")
        return None
    
    # Create prediction function with loaded model
    def predict_fn(image1, image2):
        return predict_similarity(image1, image2, model, checkpoint)
    
    # Create Gradio interface
    with gr.Blocks(title="Contrastive Similarity Classifier") as demo:
        gr.Markdown("""
        # DinoV3 + Contrastive Learning Binary Classifier for Similar Imagery
        
        **Model Details:**
        - Backbone: DinoV3 with contrastive learning
        - Uses distance-based similarity with optimized threshold
        - Same = distance < threshold
        - Different = distance >= threshold
        """)
        
        with gr.Row():
            with gr.Column():
                image1 = gr.Image(
                    label="First Image", 
                    type="pil",
                    height=300
                )
                
            with gr.Column():
                image2 = gr.Image(
                    label="Second Image", 
                    type="pil", 
                    height=300
                )
        
        with gr.Row():
            predict_btn = gr.Button("🔍 Compare Images", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        with gr.Row():
            with gr.Column():
                result_text = gr.Textbox(
                    label="Prediction Result",
                    lines=3,
                    interactive=False
                )
                
            with gr.Column():
                similarity_score = gr.Number(
                    label="Similarity Score",
                    precision=3,
                    interactive=False
                )
        
        # Model info
        gr.Markdown(f"""
        ### Model Information:
        - **Architecture**: {checkpoint['model_config']['model_name']}
        - **Embedding Dimension**: {checkpoint['model_config']['embedding_dim']}
        - **Contrastive Margin**: {checkpoint['model_config']['margin']}
        - **Optimal Threshold**: {checkpoint['final_metrics']['optimal_threshold']:.4f}
        - **Final Accuracy**: {checkpoint['final_metrics']['accuracy']:.3f}
        - **Final F1 Score**: {checkpoint['final_metrics']['f1_score']:.3f}
        """)
        
        # Event handlers
        predict_btn.click(
            fn=predict_fn,
            inputs=[image1, image2],
            outputs=[result_text, similarity_score]
        )
        
        clear_btn.click(
            fn=lambda: (None, None, "", 0.0),
            outputs=[image1, image2, result_text, similarity_score]
        )
        
        # Example images (if available)
        gr.Markdown("""
        ### Tips:
        - Upload two images to compare their similarity
        - The model uses contrastive learning to measure distance between image embeddings
        - Lower distances indicate more similar images
        - The optimal threshold was determined during training
        """)
    
    return demo

def main():
    """Main function to launch Gradio interface"""
    print("=" * 60)
    print("CONTRASTIVE SIMILARITY CLASSIFIER - GRADIO INTERFACE")
    print("=" * 60)
    
    # Check if model file exists
    model_path = "change_detection_contrastive_classifier.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please run the training script (main.py) first to generate the model.")
        return
    
    # Create and launch interface
    demo = create_gradio_interface()
    
    if demo is not None:
        print("Launching Gradio interface...")
        demo.launch(
            share=False,  # Set to True to create public link
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,  # Default Gradio port
            show_error=True
        )
    else:
        print("Failed to create Gradio interface")

if __name__ == "__main__":
    main()