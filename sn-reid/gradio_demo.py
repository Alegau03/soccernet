#!/usr/bin/env python3
"""

This script provides an interactive demonstration of our best performing
Re-ID model (OsNet-AIN) with two modes of operation:

1. GRADIO WEB INTERFACE: Launch an interactive web application where users
    can select query images and visualize re-identification results in 
    real-time with color-coded accuracy indicators.
    
2. BATCH IMAGE GENERATION: Save visualization images to disk for use in
    reports, presentations, or publications. Includes "smart selection"
    that prioritizes queries with correct matches for better examples.

Key Features:
    - Uses OsNet-AIN (best single model: mAP 56.83%, Rank-1 43.64%)
    - Generates publication-quality visualization images
    - Color-coded results: Blue=Query, Green=Correct, Red=Incorrect
    - Smart query selection for finding illustrative examples

Usage:
    # Launch interactive Gradio interface
    python gradio_demo.py
    
    # Save 10 visualization images without launching Gradio
    python gradio_demo.py --save-samples 10 --no-gradio
    
    # Save to custom directory
    python gradio_demo.py --save-samples 20 --output-dir my_visualizations

Dependencies:
    - gradio: Web interface framework
    - torch, torchvision: Deep learning framework
    - torchreid: Person re-identification library
    - Pillow (PIL): Image processing

================================================================================
"""

import gradio as gr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchreid.utils import read_image
from torchreid.data.datasets.image.soccernetv3 import Soccernetv3
import torchreid
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime


# ==============================================================================
#                           CONFIGURATION
# ==============================================================================

# Path to the trained OsNet-AIN checkpoint
MODEL_PATH = 'final_models/OsNet.tar'

# Architecture name for model building (must match training configuration)
MODEL_ARCH = 'osnet_ain_x1_0'

# Number of top results to display in visualizations
TOP_K = 10

# Default output directory for saved visualization images
OUTPUT_DIR = 'visualization_output'


# ==============================================================================
#                           DATASET UTILITIES
# ==============================================================================

class SimpleDataset(Dataset):
    """
    Lightweight wrapper for SoccerNet ReID data.
    
    Converts raw data tuples into a format suitable for PyTorch DataLoaders
    and applies preprocessing transformations during data loading.
    
    Attributes:
        data (list): List of (image_path, person_id, camera_id) tuples
        transform (callable): Image preprocessing transformations
    """
    
    def __init__(self, data, transform=None):
        """
        Initialize the dataset wrapper.
        
        Args:
            data: Raw data from SoccerNet (query or gallery split)
            transform: torchvision transforms to apply
        """
        self.data = data
        self.transform = transform
        
    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)
        
    def __getitem__(self, index):
        """
        Load and preprocess a single sample.
        
        Includes error handling for corrupted or missing images,
        returning a zero tensor as fallback to prevent crashes.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Contains 'img', 'pid', 'camid', 'img_path'
        """
        item = self.data[index]
        img_path, pid, camid = item[0], item[1], item[2]
        
        try:
            # Load image using torchreid's utility function
            img = read_image(img_path)
            if self.transform:
                img = self.transform(img)
        except Exception:
            # Fallback for corrupted images - return zero tensor
            img = torch.zeros((3, 256, 128))
            
        return {
            'img': img, 
            'pid': pid, 
            'camid': camid, 
            'img_path': img_path
        }


def build_transforms():
    """
    Build standard image preprocessing pipeline.
    
    Uses ImageNet normalization statistics since models are pretrained
    on ImageNet. Resolution 256x128 follows standard ReID conventions
    (height > width for person images).
    
    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    return T.Compose([
        T.Resize((256, 128)),     # Standard ReID resolution (height x width)
        T.ToTensor(),             # Convert to tensor, scale to [0, 1]
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        ),
    ])


# ==============================================================================
#                           GLOBAL STATE MANAGER
# ==============================================================================

class DemoState:
    """
    Global state container for the demo application.
    
    Stores pre-computed features and distance matrices to enable
    fast lookup during interactive use. Initializing these once
    avoids repeated expensive computations.
    
    Attributes:
        device (torch.device): Compute device (CPU/GPU/MPS)
        model (nn.Module): Loaded OsNet-AIN model
        q_feats (Tensor): Pre-extracted query feature vectors
        g_feats (Tensor): Pre-extracted gallery feature vectors
        q_pids (ndarray): Query person IDs
        g_pids (ndarray): Gallery person IDs
        q_paths (ndarray): Query image file paths
        g_paths (ndarray): Gallery image file paths
        dist_mat (ndarray): Pre-computed query-gallery distance matrix
        ready (bool): Flag indicating initialization complete
    """
    
    def __init__(self):
        """Initialize empty state - populated by load_model() and extract_all_features()."""
        self.device = None
        self.model = None
        self.q_feats = None
        self.g_feats = None
        self.q_pids = None
        self.g_pids = None
        self.q_paths = None
        self.g_paths = None
        self.dist_mat = None
        self.ready = False

# Global state instance - shared across all functions
state = DemoState()


# ==============================================================================
#                           MODEL LOADING
# ==============================================================================

def load_model():
    """
    Load the OsNet-AIN model from checkpoint.
    
    This function:
    1. Selects the best available compute device (MPS > CUDA > CPU)
    2. Builds the OsNet-AIN architecture
    3. Loads pre-trained weights from checkpoint file
    4. Sets model to evaluation mode
    
    The model checkpoint should contain 'state_dict' and optionally
    'epoch' and 'rank1' for logging purposes.
    
    Raises:
        FileNotFoundError: If MODEL_PATH does not exist
    """
    print("=" * 50)
    print("Loading OsNet-AIN Model...")
    print("=" * 50)
    
    # ========== Device Selection ==========
    # Prioritize Apple Silicon (MPS), then NVIDIA GPU, finally CPU
    if torch.backends.mps.is_available():
        state.device = torch.device('mps')
    elif torch.cuda.is_available():
        state.device = torch.device('cuda')
    else:
        state.device = torch.device('cpu')
    print(f"Using device: {state.device}")
    
    # Verify checkpoint exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    # ========== Build Model Architecture ==========
    # OsNet-AIN: Omni-Scale Network with Attentive Instance Normalization
    model = torchreid.models.build_model(
        name=MODEL_ARCH, 
        num_classes=1000,      # Placeholder, we only use feature extractor
        pretrained=False       # We'll load our own weights
    )
    model.to(state.device)
    
    # ========== Load Checkpoint Weights ==========
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    sd = checkpoint.get('state_dict', checkpoint)
    
    # Clean state dict:
    # - Remove 'module.' prefix (from DataParallel training)
    # - Remove classifier weights (not needed for feature extraction)
    clean_sd = {
        k.replace('module.', ''): v 
        for k, v in sd.items() 
        if 'classifier' not in k
    }
    
    model.load_state_dict(clean_sd, strict=False)
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    
    state.model = model
    
    # Log training information if available
    print(f" Model loaded: {MODEL_ARCH}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    if checkpoint.get('rank1'):
        print(f"   Training Rank-1: {checkpoint.get('rank1'):.2%}")


# ==============================================================================
#                           FEATURE EXTRACTION
# ==============================================================================

def extract_all_features():
    """
    Extract feature embeddings for all query and gallery images.
    
    This is the main initialization function that:
    1. Loads the SoccerNet validation dataset
    2. Extracts L2-normalized feature vectors for all images
    3. Pre-computes the full query-gallery distance matrix
    
    Features are cached in the global state for fast lookup during
    interactive demo use. Distance matrix computation uses cosine
    similarity: distance = 1 - dot_product(q, g).
    
    Note:
        This function may take several minutes on first run due to
        processing ~46,000 images. Progress is printed to console.
    """
    print("\nLoading dataset...")
    
    # Load SoccerNet dataset (validation split)
    dataset = Soccernetv3(root='datasets', soccernetv3_training_subset=1.0)
    transform = build_transforms()
    
    # Create DataLoaders for efficient batch processing
    q_loader = DataLoader(
        SimpleDataset(dataset.query, transform), 
        batch_size=64, 
        shuffle=False, 
        num_workers=0  # MPS doesn't support multiprocessing
    )
    g_loader = DataLoader(
        SimpleDataset(dataset.gallery, transform), 
        batch_size=64, 
        shuffle=False, 
        num_workers=0
    )
    
    # Storage for extracted data
    q_feats, q_pids, q_paths = [], [], []
    g_feats, g_pids, g_paths = [], [], []
    
    # ========== Extract Query Features ==========
    print("Extracting Query features...")
    with torch.no_grad():  # Disable gradient computation for inference
        for i, batch in enumerate(q_loader):
            # Progress indicator
            if i % 20 == 0: 
                print(f"  Batch {i}/{len(q_loader)}", end='\r')
            
            imgs = batch['img'].to(state.device)
            
            # Forward pass through model
            f = state.model(imgs)
            
            # Handle models that return (features, logits) tuple
            if isinstance(f, (tuple, list)): 
                f = f[-1]
            
            # L2-normalize for cosine similarity computation
            f = F.normalize(f, p=2, dim=1)
            
            # Store on CPU to save GPU memory
            q_feats.append(f.cpu())
            q_pids.extend(batch['pid'])
            q_paths.extend(batch['img_path'])
    
    # ========== Extract Gallery Features ==========
    print("\nExtracting Gallery features...")
    with torch.no_grad():
        for i, batch in enumerate(g_loader):
            if i % 20 == 0: 
                print(f"  Batch {i}/{len(g_loader)}", end='\r')
            
            imgs = batch['img'].to(state.device)
            f = state.model(imgs)
            if isinstance(f, (tuple, list)): 
                f = f[-1]
            f = F.normalize(f, p=2, dim=1)
            
            g_feats.append(f.cpu())
            g_pids.extend(batch['pid'])
            g_paths.extend(batch['img_path'])
    
    # ========== Store in Global State ==========
    state.q_feats = torch.cat(q_feats, dim=0)
    state.g_feats = torch.cat(g_feats, dim=0)
    state.q_pids = np.asarray(q_pids)
    state.g_pids = np.asarray(g_pids)
    state.q_paths = np.asarray(q_paths)
    state.g_paths = np.asarray(g_paths)
    
    # ========== Pre-compute Distance Matrix ==========
    # Using cosine distance: d = 1 - cos_similarity = 1 - dot_product
    # For L2-normalized vectors, dot product equals cosine similarity
    print("\nComputing distance matrix...")
    state.dist_mat = (1 - torch.mm(state.q_feats, state.g_feats.t())).numpy()
    
    print(f"\nReady! Query: {len(q_pids)}, Gallery: {len(g_pids)}")
    state.ready = True


# ==============================================================================
#                           VISUALIZATION GENERATION
# ==============================================================================

def create_visualization_image(query_idx, save_path=None):
    """
    Create a single visualization image showing query and top-K gallery matches.
    
    The visualization displays:
    - Query image on the left with BLUE border
    - Top-K gallery images on the right, each with:
      - GREEN border if correct match (same person ID)
      - RED border if incorrect match (different person ID)
    - Rank labels below each gallery image
    
    This format is ideal for publications and presentations as it provides
    an immediate visual understanding of model performance.
    
    Args:
        query_idx (int): Index of the query image in state.q_paths
        save_path (str, optional): If provided, save image to this path
        
    Returns:
        PIL.Image: The generated visualization image, or None if not ready
    """
    if not state.ready:
        return None
    
    # Get query information
    query_path = state.q_paths[query_idx]
    query_pid = state.q_pids[query_idx]
    
    # Retrieve pre-computed distances and sort to get top-K
    distances = state.dist_mat[query_idx]
    sorted_indices = np.argsort(distances)[:TOP_K]
    
    # ========== Load Images ==========
    try:
        query_img = Image.open(query_path).convert('RGB')
    except:
        # Fallback for missing/corrupted images
        query_img = Image.new('RGB', (128, 256), color='gray')
    
    # Load gallery images and determine correctness
    gallery_imgs = []
    for g_idx in sorted_indices:
        g_path = state.g_paths[g_idx]
        g_pid = state.g_pids[g_idx]
        is_correct = (g_pid == query_pid)  # Match if same person ID
        
        try:
            g_img = Image.open(g_path).convert('RGB')
        except:
            g_img = Image.new('RGB', (128, 256), color='gray')
            
        gallery_imgs.append((g_img, is_correct, g_pid, distances[g_idx]))
    
    # ========== Create Canvas ==========
    img_height = 280   # Height for image + label
    img_width = 140    # Width for image + padding
    border_width = 4   # Border thickness in pixels
    
    # Calculate total canvas size
    total_width = img_width + 30 + (img_width * TOP_K)  # Query + separator + gallery
    total_height = img_height + 40  # Extra space for title
    
    result = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(result)
    
    # ========== Load Fonts ==========
    try:
        # Use Helvetica on macOS
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        # Fallback to PIL default font
        font = ImageFont.load_default()
        font_small = font
        font_title = font
    
    # ========== Draw Title ==========
    draw.text((10, 5), f"Query ID: {query_pid}", fill='black', font=font_title)
    
    y_offset = 30  # Start below title
    
    # ========== Draw Query Image (Blue Border) ==========
    query_resized = query_img.resize((128, 256))
    x_pos = 5
    
    # Draw blue border rectangle
    draw.rectangle(
        [x_pos - border_width, y_offset - border_width, 
         x_pos + 128 + border_width, y_offset + 256 + border_width], 
        outline='blue', 
        width=border_width
    )
    result.paste(query_resized, (x_pos, y_offset))
    draw.text((x_pos, y_offset + 260), "QUERY", fill='blue', font=font)
    
    # ========== Draw Gallery Images (Green/Red Borders) ==========
    x_start = img_width + 30  # Start position for gallery
    
    for rank, (g_img, is_correct, g_pid, dist) in enumerate(gallery_imgs):
        x_pos = x_start + rank * img_width
        
        # Resize gallery image
        g_resized = g_img.resize((128, 256))
        
        # Draw colored border based on correctness
        border_color = 'green' if is_correct else 'red'
        draw.rectangle(
            [x_pos - border_width, y_offset - border_width, 
             x_pos + 128 + border_width, y_offset + 256 + border_width], 
            outline=border_color, 
            width=border_width
        )
        
        result.paste(g_resized, (x_pos, y_offset))
        
        # Draw rank label with checkmark/cross
        status = "✓" if is_correct else "✗"
        draw.text(
            (x_pos, y_offset + 260), 
            f"R{rank + 1} {status}", 
            fill=border_color, 
            font=font_small
        )
    
    # ========== Save if Path Provided ==========
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.save(save_path, quality=95)
        print(f"Saved: {save_path}")
    
    return result


def find_good_query_indices(min_correct=1, max_samples=100):
    """
    Find query indices that have successful matches in top-K results.
    
    This "smart selection" function identifies queries where the model
    performs well, which are better for creating illustrative examples
    in reports and presentations.
    
    Args:
        min_correct (int): Minimum number of correct matches required
        max_samples (int): Maximum number of indices to return
        
    Returns:
        list: Query indices sorted by number of correct matches (descending)
    """
    good_indices = []
    
    for idx in range(len(state.q_pids)):
        query_pid = state.q_pids[idx]
        distances = state.dist_mat[idx]
        sorted_indices = np.argsort(distances)[:TOP_K]
        
        # Count how many correct matches appear in top-K
        correct_count = sum(
            1 for g_idx in sorted_indices 
            if state.g_pids[g_idx] == query_pid
        )
        
        if correct_count >= min_correct:
            good_indices.append((idx, correct_count))
    
    # Sort by correctness count (best examples first)
    good_indices.sort(key=lambda x: x[1], reverse=True)
    
    return [idx for idx, _ in good_indices[:max_samples]]


def save_visualization_samples(n_samples=10, output_dir=OUTPUT_DIR, smart_select=True):
    """
    Save multiple visualization images to disk.
    
    This function generates visualization images suitable for reports
    and presentations. With smart_select=True, it prioritizes queries
    that have correct matches, producing more illustrative examples.
    
    Args:
        n_samples (int): Number of images to save
        output_dir (str): Directory to save images
        smart_select (bool): If True, prioritize queries with correct matches
        
    Returns:
        list: Paths to saved images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving {n_samples} visualization samples to {output_dir}/...")
    
    if smart_select:
        # ========== Smart Selection Mode ==========
        print("  Using smart selection (queries with correct matches)...")
        good_indices = find_good_query_indices(min_correct=1, max_samples=n_samples * 3)
        
        if len(good_indices) >= n_samples:
            # Randomly sample from good queries for variety
            indices = np.random.choice(
                good_indices, 
                min(n_samples, len(good_indices)), 
                replace=False
            )
        else:
            # Fall back: use all good indices + some random ones
            remaining = n_samples - len(good_indices)
            random_indices = [
                i for i in range(len(state.q_paths)) 
                if i not in good_indices
            ]
            indices = good_indices + list(np.random.choice(
                random_indices,
                min(remaining, len(random_indices)),
                replace=False
            ))
        
        print(f"  Found {len(good_indices)} queries with ≥1 correct match in top-{TOP_K}")
    else:
        # ========== Random Selection Mode ==========
        indices = np.random.choice(
            len(state.q_paths), 
            min(n_samples, len(state.q_paths)), 
            replace=False
        )
    
    # ========== Generate and Save Images ==========
    saved_paths = []
    for i, idx in enumerate(indices):
        filename = f"reid_result_{i+1:02d}_pid{state.q_pids[idx]}.png"
        save_path = os.path.join(output_dir, filename)
        create_visualization_image(idx, save_path)
        saved_paths.append(save_path)
    
    print(f"Saved {len(saved_paths)} images to {output_dir}/")
    return saved_paths


# ==============================================================================
#                           GRADIO WEB INTERFACE
# ==============================================================================

def search_by_index(query_idx):
    """
    Retrieve top-K matches for a given query index.
    
    Args:
        query_idx (int): Index of query in state.q_paths
        
    Returns:
        tuple: (query_path, query_pid, results_list)
            where results_list contains dicts with 'path', 'rank', 
            'correct', 'distance', 'pid' for each gallery match
    """
    if not state.ready or query_idx is None:
        return None, []
    
    query_idx = int(query_idx)
    query_path = state.q_paths[query_idx]
    query_pid = state.q_pids[query_idx]
    
    # Get sorted gallery matches
    distances = state.dist_mat[query_idx]
    sorted_indices = np.argsort(distances)[:TOP_K]
    
    # Build results list
    results = []
    for rank, g_idx in enumerate(sorted_indices):
        g_path = state.g_paths[g_idx]
        g_pid = state.g_pids[g_idx]
        is_correct = (g_pid == query_pid)
        
        results.append({
            'path': g_path,
            'rank': rank + 1,
            'correct': is_correct,
            'distance': distances[g_idx],
            'pid': g_pid
        })
    
    return query_path, query_pid, results


def demo_search(query_selector):
    """
    Main search handler for Gradio interface.
    
    Parses the dropdown selection to extract query index,
    then generates and returns the visualization image.
    
    Args:
        query_selector (str): Dropdown value in format "filename (index)"
        
    Returns:
        PIL.Image: Visualization image for display in Gradio
    """
    if query_selector is None or query_selector == "":
        return None
    
    # Parse index from selector format: "image_name.png (123)"
    try:
        idx = int(query_selector.split("(")[-1].rstrip(")"))
    except:
        return None
    
    return create_visualization_image(idx)


def create_gallery_results_image(query_idx):
    """
    Create an image showing ONLY the top-K gallery results (without query).
    
    For the Gradio interface: displays only the gallery matches with
    green/red borders based on correctness.
    
    Args:
        query_idx (int): Index of the query image
        
    Returns:
        PIL.Image: Gallery results visualization
    """
    if not state.ready:
        return None
    
    query_pid = state.q_pids[query_idx]
    
    # Get top-K gallery matches
    distances = state.dist_mat[query_idx]
    sorted_indices = np.argsort(distances)[:TOP_K]
    
    # Load gallery images
    gallery_imgs = []
    for g_idx in sorted_indices:
        g_path = state.g_paths[g_idx]
        g_pid = state.g_pids[g_idx]
        is_correct = (g_pid == query_pid)
        
        try:
            g_img = Image.open(g_path).convert('RGB')
        except:
            g_img = Image.new('RGB', (128, 256), color='gray')
            
        gallery_imgs.append((g_img, is_correct, g_pid, distances[g_idx]))
    
    # Create canvas for gallery images only
    img_width = 145
    border_width = 8  # Thick border for visibility
    
    total_width = img_width * TOP_K + 20
    total_height = 310  # Image + labels
    
    result = Image.new('RGB', (total_width, total_height), color='white')
    draw = ImageDraw.Draw(result)
    
    # Load fonts
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
    
    y_offset = 10
    
    # Draw each gallery image with colored border
    for rank, (g_img, is_correct, g_pid, dist) in enumerate(gallery_imgs):
        x_pos = 10 + rank * img_width
        
        g_resized = g_img.resize((128, 256))
        
        # Green = correct, Red = incorrect
        border_color = '#00FF00' if is_correct else 'red'
        draw.rectangle(
            [x_pos - border_width, y_offset - border_width, 
             x_pos + 128 + border_width, y_offset + 256 + border_width], 
            outline=border_color, 
            width=border_width
        )
        
        result.paste(g_resized, (x_pos, y_offset))
        
        # Rank label
        status = "✓" if is_correct else "✗"
        draw.text((x_pos + 5, y_offset + 262), f"R{rank+1} {status}", fill=border_color, font=font)
    
    return result


def show_query_image(query_selector):
    """
    Show the selected query image immediately when dropdown changes.
    
    Args:
        query_selector (str): Dropdown value in format "filename (index)"
        
    Returns:
        PIL.Image: Query image with blue border, or None if invalid
    """
    if query_selector is None or query_selector == "":
        return None
    
    try:
        idx = int(query_selector.split("(")[-1].rstrip(")"))
    except:
        return None
    
    query_path = state.q_paths[idx]
    query_pid = state.q_pids[idx]
    
    try:
        query_img = Image.open(query_path).convert('RGB')
    except:
        return None
    
    # Create image with blue border and label
    border_width = 4
    query_resized = query_img.resize((128, 256))
    
    result = Image.new('RGB', (150, 310), color='white')
    draw = ImageDraw.Draw(result)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
    
    x_pos, y_offset = 10, 10
    
    draw.rectangle(
        [x_pos - border_width, y_offset - border_width, 
         x_pos + 128 + border_width, y_offset + 256 + border_width], 
        outline='blue', 
        width=border_width
    )
    result.paste(query_resized, (x_pos, y_offset))
    draw.text((x_pos + 5, y_offset + 262), f"Query (ID: {query_pid})", fill='blue', font=font)
    
    return result


def demo_search_gallery(query_selector):
    """
    Search function for Gradio - returns ONLY gallery results image.
    
    Args:
        query_selector (str): Dropdown value
        
    Returns:
        PIL.Image: Gallery results with green/red borders
    """
    if query_selector is None or query_selector == "":
        return None
    
    try:
        idx = int(query_selector.split("(")[-1].rstrip(")"))
    except:
        return None
    
    return create_gallery_results_image(idx)


def get_query_choices():
    """
    Generate dropdown choices for query selection.
    
    Filters to ONLY include queries that have at least one correct match
    in the top-10 results. Sorted by best rank position (Rank-1 first).
    
    Returns:
        list: Dropdown option strings sorted by best rank
    """
    if not state.ready:
        return []
    
    # Find queries with correct matches in top-10 and their best rank
    good_queries = []
    
    for idx in range(len(state.q_pids)):
        query_pid = state.q_pids[idx]
        distances = state.dist_mat[idx]
        sorted_indices = np.argsort(distances)[:TOP_K]
        
        # Find the best (lowest) rank with a correct match
        best_rank = None
        for rank, g_idx in enumerate(sorted_indices):
            if state.g_pids[g_idx] == query_pid:
                best_rank = rank + 1  # 1-indexed
                break
        
        if best_rank is not None:
            good_queries.append((idx, best_rank))
    
    # Sort by best rank (ascending: Rank-1 first, then Rank-2, etc.)
    good_queries.sort(key=lambda x: x[1])
    
    print(f"Found {len(good_queries)} queries with correct matches in top-{TOP_K}")
    
    # Format for dropdown: show rank info
    return [
        f"[R{best_rank}] {os.path.basename(state.q_paths[idx])} ({idx})" 
        for idx, best_rank in good_queries
    ]


def create_app():
    """
    Build the Gradio web application.
    
    Creates an interactive interface with:
    - Markdown header with model information
    - Dropdown for query selection
    - Refresh button to load new random queries
    - Search button to execute re-identification
    - Image display for visualization results
    
    Returns:
        gr.Blocks: Configured Gradio application
    """
    
    with gr.Blocks(
        title="SoccerNet ReID Demo - OsNet", 
        theme=gr.themes.Soft()
    ) as app:
        
        # ========== Header ==========
        gr.Markdown("""
        # ⚽ SoccerNet Person Re-Identification Demo
        
        **Model**: OsNet-AIN (Best Single Model - mAP 56.83%, Rank-1 43.64%)
        
        1. Select a query image from the dropdown → it will appear on the left
        2. Click **Search** → see top-10 matches on the right
        
        🟢 **Green border** = Correct match (same person) | 🔴 **Red border** = Wrong match
        """)
        
        # ========== Controls Row ==========
        with gr.Row():
            query_dropdown = gr.Dropdown(
                choices=get_query_choices() if state.ready else [],
                label="📷 Select Query Image",
                info="Choose a query image to search",
                interactive=True,
                scale=3
            )
            search_btn = gr.Button("🔍 Search", variant="primary", size="lg", scale=1)
        
        # ========== Image Display Row ==========
        with gr.Row():
            # Query image on the left (smaller)
            with gr.Column(scale=1):
                query_image = gr.Image(
                    label="Query Image",
                    type="pil",
                    height=320,
                    show_label=True
                )
            
            # Gallery results on the right (larger)
            with gr.Column(scale=4):
                gallery_image = gr.Image(
                    label="Top-10 Re-Identification Results",
                    type="pil",
                    height=320,
                    show_label=True
                )
        
        # ========== Event Handlers ==========
        # Show query image when dropdown selection changes
        query_dropdown.change(
            fn=show_query_image,
            inputs=[query_dropdown],
            outputs=[query_image]
        )
        
        # Show gallery results when Search is clicked
        search_btn.click(
            fn=demo_search_gallery,
            inputs=[query_dropdown],
            outputs=[gallery_image]
        )
        

        
        # ========== Footer ==========
        gr.Markdown("""
        ---
        **Dataset**: SoccerNet ReID v3 Validation Set (11,638 queries, 34,355 gallery images)
        """)
    
    return app


# ==============================================================================
#                           MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    # ========== Command Line Arguments ==========
    parser = argparse.ArgumentParser(
        description='SoccerNet ReID Interactive Visualization Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web interface
  python gradio_demo.py
  
  # Save 10 example images for report
  python gradio_demo.py --save-samples 10 --no-gradio
  
  # Save to custom directory with more samples
  python gradio_demo.py --save-samples 20 --output-dir report_images --no-gradio
        """
    )
    
    parser.add_argument(
        '--save-samples', 
        type=int, 
        default=0, 
        help='Number of visualization images to save (0 = skip)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=OUTPUT_DIR,
        help='Output directory for saved images'
    )
    parser.add_argument(
        '--no-gradio', 
        action='store_true',
        help='Save images only, do not launch Gradio web interface'
    )
    
    args = parser.parse_args()
    
    # ========== Initialization ==========
    print("=" * 50)
    print("SoccerNet ReID - OsNet Visualization Demo")
    print("=" * 50)
    
    # Load model and extract features (one-time initialization)
    load_model()
    extract_all_features()
    
    # ========== Save Samples if Requested ==========
    if args.save_samples > 0:
        save_visualization_samples(args.save_samples, args.output_dir)
    
    # ========== Launch Gradio Interface ==========
    if not args.no_gradio:
        app = create_app()
        app.launch(
            server_name="0.0.0.0",   # Listen on all interfaces
            server_port=7860,        # Default Gradio port
            share=False,             # No public URL
            show_error=True          # Show detailed errors
        )
    else:
        print("\n✅ Done! Images saved. Gradio skipped (--no-gradio).")
