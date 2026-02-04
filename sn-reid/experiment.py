#!/usr/bin/env python3
"""


This script provides a comprehensive evaluation framework for comparing 
different person re-identification strategies on the SoccerNet dataset:

1. SINGLE MODELS: Evaluate individual models (ResNet, DINOv2, OsNet)
2. ENSEMBLE METHODS: Combine multiple models using:
    - Feature Concatenation: Stack feature vectors and renormalize
    - Distance Averaging: Average distance matrices (weighted or equal)
3. RE-RANKING: Apply k-reciprocal re-ranking to improve gallery retrieval

Metrics:
    - mAP (mean Average Precision): Primary ranking metric
    - Rank-1: Probability that the correct match is at position 1
    - CMC (Cumulative Matching Characteristics): Full rank accuracy curve

Usage:
    python experiment.py --models model1.pth model2.pth --archs arch1 arch2

Output:
    - Detailed log file saved to final_models/experiment_results_<timestamp>.log
    - Console output with real-time progress

================================================================================
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchreid.utils import read_image
from torchreid.models.resnet import resnet50_fc512
from torchreid.data.datasets.image.soccernetv3 import Soccernetv3
from torchreid.metrics.rank import evaluate_rank
from torchreid.utils.rerank import re_ranking
import numpy as np
import os
from datetime import datetime


# ==============================================================================
#                           DATASET UTILITIES
# ==============================================================================

class SimpleDataset(Dataset):
    """
    Wrapper Dataset for SoccerNet ReID data.
    
    Converts the raw data tuples (image_path, person_id, camera_id) into 
    a format suitable for PyTorch DataLoaders. Applies preprocessing 
    transformations on-the-fly during data loading.
    
    Attributes:
        data (list): List of tuples containing (img_path, pid, camid)
        transform (callable): Image preprocessing transformations
    """
    
    def __init__(self, data, transform=None):
        """
        Initialize the dataset wrapper.
        
        Args:
            data: Raw data from SoccerNet dataset (query or gallery split)
            transform: torchvision transforms to apply to images
        """
        self.data = data
        self.transform = transform
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Load and preprocess a single sample.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Contains 'img' (tensor), 'pid' (person ID), 
                  'camid' (camera ID), 'img_path' (original path)
        """
        item = self.data[index]
        img_path, pid, camid = item[0], item[1], item[2]
        
        # Load image using torchreid's utility (handles various formats)
        img = read_image(img_path)
        
        # Apply preprocessing transformations
        if self.transform:
            img = self.transform(img)
            
        return {
            'img': img, 
            'pid': pid, 
            'camid': camid, 
            'img_path': img_path
        }


def build_transforms():
    """
    Build standard image preprocessing pipeline for ReID models.
    
    The transformations follow standard person ReID conventions:
    1. Resize to 256x128 (height x width) - standard ReID aspect ratio
    2. Convert to tensor (scales pixel values to [0, 1])
    3. Normalize using ImageNet statistics (models are pretrained on ImageNet)
    
    Returns:
        transforms.Compose: Composed transformation pipeline
    """
    return T.Compose([
        T.Resize((256, 128)),  # Standard ReID resolution
        T.ToTensor(),          # Convert PIL Image to Tensor
        T.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        ),
    ])


def extract_features(model, loader, device, logger):
    """
    Extract L2-normalized feature embeddings from all images in a DataLoader.
    
    This function performs inference on all images in batches, extracting
    feature vectors from the model's embedding layer. Features are L2-normalized
    to enable cosine similarity computation via dot product.
    
    Args:
        model (nn.Module): Pre-trained ReID model in eval mode
        loader (DataLoader): DataLoader containing images to process
        device (torch.device): Device to run inference on (CPU/GPU/MPS)
        logger (ExperimentLogger): Logger for progress reporting
        
    Returns:
        tuple: (features, pids, camids)
            - features (Tensor): Shape (N, D) where N=samples, D=embedding dim
            - pids (ndarray): Person IDs for each sample
            - camids (ndarray): Camera IDs for each sample
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    
    feats, pids, camids = [], [], []
    
    # Disable gradient computation for faster inference
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Progress indicator every 50 batches
            if i % 50 == 0: 
                msg = f"   Extracting batch {i}/{len(loader)}"
                print(msg, end='\r')
            
            # Move images to compute device
            img = batch['img'].to(device)
            
            # Forward pass - get feature embeddings
            f = model(img)
            
            # Some models return tuple (features, logits), extract features
            if isinstance(f, (tuple, list)):
                f = f[-1]
            
            # L2-normalize features for cosine similarity
            # This allows using dot product as similarity metric
            f = F.normalize(f, p=2, dim=1)
            
            # Store results on CPU to save GPU memory
            feats.append(f.cpu())
            pids.extend(batch['pid'])
            camids.extend(batch['camid'])
    
    print("")  # Clear the progress line
    
    # Concatenate all features into single tensor
    return torch.cat(feats, 0), np.asarray(pids), np.asarray(camids)


# ==============================================================================
#                           ENSEMBLE METHODS
# ==============================================================================

def compute_dist_matrix(q_f, g_f):
    """
    Compute squared Euclidean distance matrix between query and gallery features.
    
    For L2-normalized features, squared Euclidean distance is related to
    cosine similarity: d^2 = 2(1 - cos_sim). This function computes distances
    efficiently using the expansion: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    Args:
        q_f (Tensor): Query features, shape (M, D)
        g_f (Tensor): Gallery features, shape (N, D)
        
    Returns:
        ndarray: Distance matrix of shape (M, N) where entry [i,j] is the
                 squared Euclidean distance between query i and gallery j
    """
    m, n = q_f.size(0), g_f.size(0)
    
    # Compute ||q||^2 for all queries (expanded to match gallery size)
    # Compute ||g||^2 for all gallery (expanded to match query size)
    distmat = (
        torch.pow(q_f, 2).sum(dim=1, keepdim=True).expand(m, n) + 
        torch.pow(g_f, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    
    # Subtract 2 * q·g (the cross-term)
    distmat.addmm_(q_f, g_f.t(), beta=1, alpha=-2)
    
    return distmat.numpy()


def ensemble_feature_concat(q_feats_list, g_feats_list):
    """
    Ensemble by concatenating feature vectors from multiple models.
    
    This method creates a higher-dimensional representation by stacking
    features from all models. The concatenated features are then 
    L2-normalized before computing distances. This approach allows
    learning complementary information from different architectures.
    
    Example: 
        If Model1 produces 512-d features and Model2 produces 768-d features,
        the concatenated representation will be 1280-dimensional.
    
    Args:
        q_feats_list (list): List of query feature tensors from each model
        g_feats_list (list): List of gallery feature tensors from each model
        
    Returns:
        ndarray: Distance matrix computed from concatenated features
    """
    # Concatenate features along feature dimension
    q_cat = torch.cat(q_feats_list, dim=1)
    g_cat = torch.cat(g_feats_list, dim=1)
    
    # Re-normalize concatenated features
    q_cat = F.normalize(q_cat, p=2, dim=1)
    g_cat = F.normalize(g_cat, p=2, dim=1)
    
    return compute_dist_matrix(q_cat, g_cat)


def ensemble_dist_avg(dist_mats, weights=None):
    """
    Ensemble by averaging distance matrices with optional weighting.
    
    This late-fusion approach combines the decisions of multiple models
    at the distance/score level rather than the feature level. Weighted
    averaging allows giving more importance to stronger models.
    
    Mathematical formulation:
        D_ensemble = Σ(w_i * D_i), where Σw_i = 1
    
    Args:
        dist_mats (list): List of distance matrices from each model
        weights (list, optional): Weight for each model. If None, uses
                                  equal weights (1/N for N models)
                                  
    Returns:
        ndarray: Weighted average distance matrix
    """
    if weights is None:
        # Default to equal weighting
        weights = [1.0 / len(dist_mats)] * len(dist_mats)
    
    # Initialize with zeros
    final_dist = np.zeros_like(dist_mats[0])
    
    # Weighted sum
    for d, w in zip(dist_mats, weights):
        final_dist += w * d
    
    return final_dist


# ==============================================================================
#                           LOGGING UTILITIES
# ==============================================================================

class ExperimentLogger:
    """
    Dual-output logger for experiment tracking.
    
    Writes output to both console (for real-time monitoring) and a log file
    (for permanent record). Also accumulates results for summary table generation.
    
    Attributes:
        log_path (str): Path to the output log file
        log_file (file): Open file handle for writing
        results (list): Accumulated experiment results for summary
    """
    
    def __init__(self, log_path):
        """
        Initialize logger and create output file.
        
        Args:
            log_path (str): Path where log file will be created
        """
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')
        self.results = []  # Store (method, mAP, rank1) tuples
        
    def log(self, message, also_print=True):
        """
        Write message to log file and optionally print to console.
        
        Args:
            message (str): Message to log
            also_print (bool): If True, also print to console
        """
        self.log_file.write(message + '\n')
        self.log_file.flush()  # Ensure immediate write to disk
        if also_print:
            print(message)
    
    def log_separator(self, char='=', length=70):
        """Print a visual separator line."""
        self.log(char * length)
    
    def log_section(self, title):
        """Print a section header with surrounding separators."""
        self.log("")
        self.log_separator()
        self.log(f"  {title}")
        self.log_separator()
    
    def add_result(self, method, mAP, rank1, category=""):
        """
        Record an experiment result for final summary.
        
        Args:
            method (str): Name of the method/configuration
            mAP (float): Mean Average Precision score
            rank1 (float): Rank-1 accuracy score
            category (str): Category for grouping (e.g., "Single Model")
        """
        self.results.append({
            'Category': category,
            'Method': method,
            'mAP': mAP,
            'Rank-1': rank1
        })
        
    def close(self):
        """Close the log file."""
        self.log_file.close()


# ==============================================================================
#                           MAIN EXPERIMENT
# ==============================================================================

def run_experiments(args):
    """
    Main experiment runner.
    
    Executes a complete experiment pipeline:
    1. Setup logging and device configuration
    2. Load SoccerNet dataset (validation split)
    3. Load all specified models
    4. Extract features for each model
    5. Evaluate single model performance
    6. Test ensemble methods (concat, avg, weighted)
    7. Apply re-ranking to best ensemble
    8. Generate summary report
    
    Args:
        args: Parsed command-line arguments containing:
              - models: List of model checkpoint paths
              - archs: List of architecture names
              - gpu_id: GPU ID or '-1' for CPU
    """
    
    # ==========================================================================
    # Step 1: Setup Logging
    # ==========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiment_results_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(args.models[0]), log_filename)
    logger = ExperimentLogger(log_path)
    
    # Print experiment header
    logger.log_separator('=')
    logger.log("  SOCCERNET RE-ID: ENSEMBLE & RE-RANKING EXPERIMENT")
    logger.log_separator('=')
    logger.log(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"  Log file:  {log_path}")
    logger.log("")
    
    # ==========================================================================
    # Step 2: Configure Compute Device
    # ==========================================================================
    if args.gpu_id == '-1': 
        device = torch.device('cpu')
    else: 
        # Prefer MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    logger.log(f"  Device: {device}")

    # ==========================================================================
    # Step 3: Load Dataset
    # ==========================================================================
    logger.log_section("DATASET INFORMATION")
    logger.log("  Dataset: SoccerNet Re-ID v3")
    logger.log("  Evaluation Split: VALIDATION SET (not test set)")
    logger.log("  Subset Ratio: 100% (full validation set)")
    logger.log("")
    
    # Load SoccerNet with full training subset (affects train split only)
    dataset = Soccernetv3(root='datasets', soccernetv3_training_subset=1.0)
    
    # Log dataset statistics
    logger.log(f"  Query images:   {len(dataset.query):,} images")
    logger.log(f"  Gallery images: {len(dataset.gallery):,} images")
    logger.log(f"  Unique Query IDs: {len(set([x[1] for x in dataset.query])):,}")
    logger.log(f"  Unique Gallery IDs: {len(set([x[1] for x in dataset.gallery])):,}")
    
    # Create DataLoaders with standard preprocessing
    q_loader = DataLoader(
        SimpleDataset(dataset.query, build_transforms()), 
        batch_size=64, 
        shuffle=False, 
        num_workers=0
    )
    g_loader = DataLoader(
        SimpleDataset(dataset.gallery, build_transforms()), 
        batch_size=64, 
        shuffle=False, 
        num_workers=0
    )

    # ==========================================================================
    # Step 4: Load Models
    # ==========================================================================
    logger.log_section("MODELS CONFIGURATION")
    
    # Expand single architecture to match multiple models
    if len(args.archs) == 1 and len(args.models) > 1: 
        args.archs *= len(args.models)
    
    for i, (path, arch) in enumerate(zip(args.models, args.archs)):
        logger.log(f"  Model {i+1}: {arch}")
        logger.log(f"           Path: {path}")
    logger.log("")

    # ==========================================================================
    # Step 5: Feature Extraction
    # ==========================================================================
    logger.log_section("FEATURE EXTRACTION")
    
    q_feats_collection = []  # Query features from each model
    g_feats_collection = []  # Gallery features from each model
    pids_q, cams_q = None, None
    pids_g, cams_g = None, None

    for i, (model_path, arch) in enumerate(zip(args.models, args.archs)):
        logger.log(f"\n  [{i+1}/{len(args.models)}] Loading {arch}...")
        
        # Build model based on architecture type
        if 'resnet' in arch.lower():
            model = resnet50_fc512(num_classes=1000, pretrained=False).to(device)
        elif 'dinov2' in arch.lower() or 'dino' in arch.lower():
            import torchreid
            model = torchreid.models.build_model(
                name=arch, num_classes=161443, pretrained=True
            ).to(device)
        elif 'osnet' in arch.lower():
            import torchreid
            model = torchreid.models.build_model(
                name=arch, num_classes=1000, pretrained=False
            ).to(device)
        else:
            import torchreid
            model = torchreid.models.build_model(
                name=arch, num_classes=1000, pretrained=False
            ).to(device)
            
        # Load trained weights from checkpoint
        chk = torch.load(model_path, map_location='cpu', weights_only=False)
        sd = chk['state_dict'] if 'state_dict' in chk else chk
        
        # Clean state dict: remove 'module.' prefix and classifier weights
        clean_sd = {
            k.replace('module.', ''): v 
            for k, v in sd.items() 
            if 'classifier' not in k
        }
        model.load_state_dict(clean_sd, strict=False)
        
        # Extract features for query and gallery sets
        logger.log(f"       Extracting query features...")
        q_f, pids_q, cams_q = extract_features(model, q_loader, device, logger)
        
        logger.log(f"       Extracting gallery features...")
        g_f, pids_g, cams_g = extract_features(model, g_loader, device, logger)
        
        logger.log(f"       Feature dimension: {q_f.shape[1]}")
        
        q_feats_collection.append(q_f)
        g_feats_collection.append(g_f)

    # Helper function for evaluation
    def evaluate(dist, name, category):
        """Compute metrics and log results."""
        cmc, mAP = evaluate_rank(
            dist, pids_q, pids_g, cams_q, cams_g, 
            eval_metric='soccernetv3'
        )
        logger.add_result(name, mAP, cmc[0], category)
        return mAP, cmc[0], dist

    # ==========================================================================
    # EXPERIMENT 1: Single Model Performance
    # ==========================================================================
    logger.log_section("EXPERIMENT 1: SINGLE MODEL PERFORMANCE")
    
    dist_mats_single = []
    for i, (qf, gf) in enumerate(zip(q_feats_collection, g_feats_collection)):
        # Compute distance matrix for this model
        d = compute_dist_matrix(qf, gf)
        dist_mats_single.append(d)
        
        # Evaluate and log
        mAP, r1, _ = evaluate(d, args.archs[i], "Single Model")
        logger.log(f"  {args.archs[i]:<35} | mAP: {mAP:>6.2%} | Rank-1: {r1:>6.2%}")

    # ==========================================================================
    # EXPERIMENT 2: Ensemble Methods
    # ==========================================================================
    logger.log_section("EXPERIMENT 2: ENSEMBLE METHODS")
    
    # Method A: Feature Concatenation
    logger.log("\n  [A] Feature Concatenation (all models)")
    d_concat = ensemble_feature_concat(q_feats_collection, g_feats_collection)
    mAP, r1, _ = evaluate(d_concat, "Feature Concatenation", "Ensemble")
    logger.log(f"      {'Feature Concatenation':<31} | mAP: {mAP:>6.2%} | Rank-1: {r1:>6.2%}")
    
    # Method B: Distance Averaging (Equal Weights)
    logger.log("\n  [B] Distance Averaging (equal weights)")
    d_avg = ensemble_dist_avg(dist_mats_single)
    mAP, r1, _ = evaluate(d_avg, "Distance Avg (Equal)", "Ensemble")
    logger.log(f"      {'Distance Avg (Equal)':<31} | mAP: {mAP:>6.2%} | Rank-1: {r1:>6.2%}")
    
    # Method C: Weighted Distance Fusion - Grid Search
    logger.log("\n  [C] Weighted Distance Fusion (grid search)")
    
    best_w_map = 0
    best_w_config = ""
    best_dist_w = None
    
    # Define weight configurations based on number of models
    if len(dist_mats_single) == 2:
        # Two-model weight search
        for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            w2 = 1.0 - w1
            d_w = ensemble_dist_avg(dist_mats_single, weights=[w1, w2])
            cmc, mAP = evaluate_rank(
                d_w, pids_q, pids_g, cams_q, cams_g, 
                eval_metric='soccernetv3'
            )
            logger.log(f"      w=({w1:.1f}, {w2:.1f}) -> mAP: {mAP:>6.2%}")
            if mAP > best_w_map:
                best_w_map, best_w_config = mAP, f"({w1:.1f}, {w2:.1f})"
                best_dist_w, best_r1 = d_w, cmc[0]
                
    elif len(dist_mats_single) == 3:
        # Three-model weight configurations
        weight_configs = [
            (0.33, 0.33, 0.34),  # Equal weights
            (0.5, 0.25, 0.25),   # Emphasize Model 1
            (0.25, 0.5, 0.25),   # Emphasize Model 2
            (0.25, 0.25, 0.5),   # Emphasize Model 3
            (0.4, 0.4, 0.2),     
            (0.4, 0.2, 0.4),     
            (0.2, 0.4, 0.4),     
            (0.6, 0.2, 0.2),     
            (0.2, 0.6, 0.2),     
            (0.2, 0.2, 0.6),     # Heavy weight on Model 3 (typically OsNet)
        ]
        
        for w1, w2, w3 in weight_configs:
            d_w = ensemble_dist_avg(dist_mats_single, weights=[w1, w2, w3])
            cmc, mAP = evaluate_rank(
                d_w, pids_q, pids_g, cams_q, cams_g, 
                eval_metric='soccernetv3'
            )
            logger.log(f"      w=({w1:.2f}, {w2:.2f}, {w3:.2f}) -> mAP: {mAP:>6.2%}")
            if mAP > best_w_map:
                best_w_map, best_w_config = mAP, f"({w1:.2f}, {w2:.2f}, {w3:.2f})"
                best_dist_w, best_r1 = d_w, cmc[0]
    
    # Log best weighted configuration
    if best_dist_w is not None:
        logger.add_result(f"Weighted Best {best_w_config}", best_w_map, best_r1, "Ensemble")
        logger.log(f"\n      ★ BEST WEIGHTED: {best_w_config} -> mAP: {best_w_map:>6.2%} | Rank-1: {best_r1:>6.2%}")
        target_dist_for_rr = best_dist_w
        target_name_for_rr = f"Weighted {best_w_config}"
    else:
        target_dist_for_rr = d_concat
        target_name_for_rr = "Feature Concat"

    # ==========================================================================
    # EXPERIMENT 3: Re-Ranking
    # ==========================================================================
    logger.log_section("EXPERIMENT 3: RE-RANKING")
    logger.log(f"  Applying re-ranking to best ensemble: {target_name_for_rr}")
    logger.log("")
    
    # Prepare query-query and gallery-gallery distance matrices for re-ranking
    # These are needed for k-reciprocal neighbor computation
    q_cat = torch.cat(q_feats_collection, dim=1)
    q_cat = F.normalize(q_cat, p=2, dim=1)
    g_cat = torch.cat(g_feats_collection, dim=1)
    g_cat = F.normalize(g_cat, p=2, dim=1)
    
    q_q = compute_dist_matrix(q_cat, q_cat)  # Query-Query distances
    g_g = compute_dist_matrix(g_cat, g_cat)  # Gallery-Gallery distances
    
    # Re-ranking configurations to test
    rr_configs = [
        {
            'k1': 20, 'k2': 6, 'lambda_value': 0.3, 
            'name': 'Standard (k1=20, k2=6)'
        },
        {
            'k1': 10, 'k2': 3, 'lambda_value': 0.3, 
            'name': 'Aggressive (k1=10, k2=3)'
        },
        {
            'k1': 60, 'k2': 10, 'lambda_value': 0.3, 
            'name': 'Broad (k1=60, k2=10)'
        },
        {
            'k1': 20, 'k2': 6, 'lambda_value': 0.5, 
            'name': 'Lambda=0.5 (k1=20, k2=6)'
        },
    ]
    
    # Test each re-ranking configuration
    for cfg in rr_configs:
        logger.log(f"  Testing: {cfg['name']}", also_print=True)
        
        # Apply k-reciprocal re-ranking
        d_rr = re_ranking(
            target_dist_for_rr, q_q, g_g,
            k1=cfg['k1'], 
            k2=cfg['k2'], 
            lambda_value=cfg['lambda_value']
        )
        
        mAP, r1, _ = evaluate(d_rr, f"Re-Rank: {cfg['name']}", "Re-Ranking")
        logger.log(f"      mAP: {mAP:>6.2%} | Rank-1: {r1:>6.2%}")

    # ==========================================================================
    # FINAL SUMMARY TABLE
    # ==========================================================================
    logger.log_section("FINAL SUMMARY")
    
    categories = ["Single Model", "Ensemble", "Re-Ranking"]
    
    # Print formatted table
    header = f"  {'Method':<45} | {'mAP':>8} | {'Rank-1':>8}"
    logger.log(header)
    logger.log("  " + "-" * 68)
    
    for cat in categories:
        cat_results = [r for r in logger.results if r['Category'] == cat]
        if cat_results:
            logger.log(f"  [{cat}]")
            for r in cat_results:
                logger.log(
                    f"    {r['Method']:<43} | {r['mAP']:>7.2%} | {r['Rank-1']:>7.2%}"
                )
            logger.log("")
    
    # Highlight overall best configuration
    best = max(logger.results, key=lambda x: x['mAP'])
    logger.log_separator()
    logger.log(f"  ★ BEST CONFIGURATION:")
    logger.log(f"    Method:  {best['Method']}")
    logger.log(f"    mAP:     {best['mAP']:.2%}")
    logger.log(f"    Rank-1:  {best['Rank-1']:.2%}")
    logger.log_separator()
    
    # Finalize
    logger.log(f"\n  Results saved to: {log_path}")
    logger.close()
    
    print(f"\n✅ Experiment complete! Results saved to: {log_path}")


# ==============================================================================
#                           COMMAND LINE INTERFACE
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SoccerNet ReID: Ensemble & Re-Ranking Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model evaluation
  python experiment.py --models final_models/OsNet.tar --archs osnet_ain_x1_0
  
  # Three model ensemble
  python experiment.py \\
    --models final_models/ResNet.pth final_models/DINOv2.pth final_models/OsNet.tar \\
    --archs resnet50_fc512 dinov2_vitb14_lora osnet_ain_x1_0
        """
    )
    
    parser.add_argument(
        '--models', 
        nargs='+', 
        required=True, 
        help='Paths to model checkpoint files (.pth or .tar)'
    )
    parser.add_argument(
        '--archs', 
        nargs='+', 
        default=['resnet50_fc512'], 
        help='Architecture names. Options: resnet50_fc512, dinov2_vitb14_lora, osnet_ain_x1_0'
    )
    parser.add_argument(
        '--gpu-id', 
        default='0', 
        help='GPU ID to use, or -1 for CPU (default: 0)'
    )
    
    run_experiments(parser.parse_args())
