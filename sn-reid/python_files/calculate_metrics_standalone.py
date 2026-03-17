import numpy as np
import argparse
from sklearn.metrics import roc_curve
import os

def calculate_biometric_metrics(dist_matrix, pids_q, pids_g):
    """
    Standalone calculation of biometric metrics: EER, DIR@Rank1-EER, Margin.
    Args:
        dist_matrix: (M, N) np.ndarray of distances.
        pids_q: (M,) array of query person IDs.
        pids_g: (N,) array of gallery person IDs.
    """
    num_queries = dist_matrix.shape[0]
    
    # 1. Identify matches and non-matches
    # For each query, a gallery image is a match if pids_g[j] == pids_q[i]
    is_match = (pids_g[None, :] == pids_q[:, None])
    
    # Flatten everything for ROC
    y_true = is_match.flatten().astype(int)
    # Distances are dissimilarities, so we use them directly
    y_scores = dist_matrix.flatten()
    
    # 2. Compute EER
    fpr, tpr, thresholds = roc_curve(y_true, -y_scores) # negative because higher scores mean more likely match for roc_curve
    fnr = 1 - tpr
    
    # EER is point where fpr == fnr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = -thresholds[idx] # back to positive distance
    
    # 3. Compute DIR at EER threshold and Rank 1
    # Rank 1 correctness for each query
    rank1_indices = np.argmin(dist_matrix, axis=1)
    rank1_correct = (pids_g[rank1_indices] == pids_q)
    
    # Decision: match if min_dist <= eer_threshold
    min_distances = np.min(dist_matrix, axis=1)
    identified_and_correct = rank1_correct & (min_distances <= eer_threshold)
    
    # DIR(t, 1) = probability rank 1 is correct AND match score >= threshold
    # Note: our scores are distances, so match score >= threshold means distance <= threshold
    dir_eer = np.mean(identified_and_correct)
    
    # 4. Compute Margin M(t) at EER (should be ~0)
    margin_eer = np.abs(fpr[idx] - fnr[idx])
    
    # 5. Compute SRR (System Response Reliability)
    sorted_dists = np.sort(dist_matrix, axis=1)
    d1 = sorted_dists[:, 0]
    d2 = sorted_dists[:, 1]
    max_d = np.max(dist_matrix, axis=1)
    srr_scores = (d2 - d1) / (max_d + 1e-6)
    avg_srr = np.mean(srr_scores)
    
    return {
        'EER': eer,
        'EER_Threshold': eer_threshold,
        'DIR_at_EER': dir_eer,
        'Margin_at_EER': margin_eer,
        'Avg_SRR': avg_srr
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate Biometric Metrics from saved Distance Matrix and Labels")
    parser.add_argument("--dist", type=str, required=True, help="Path to distance matrix (.npy)")
    parser.add_argument("--pids_q", type=str, required=True, help="Path to query PIDs (.npy)")
    parser.add_argument("--pids_g", type=str, required=True, help="Path to gallery PIDs (.npy)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dist) or not os.path.exists(args.pids_q) or not os.path.exists(args.pids_g):
        print("Error: One or more files do not exist.")
        return
        
    dist = np.load(args.dist)
    pids_q = np.load(args.pids_q)
    pids_g = np.load(args.pids_g)
    
    print(f"Loaded distance matrix: {dist.shape}")
    print(f"Calculating metrics...")
    
    results = calculate_biometric_metrics(dist, pids_q, pids_g)
    
    print("\n" + "="*40)
    print(" BIOMETRIC METRICS SUMMARY")
    print("="*40)
    print(f" EER:            {results['EER']*100:.2f}%")
    print(f" EER Threshold:  {results['EER_Threshold']:.4f}")
    print(f" DIR at EER:     {results['DIR_at_EER']*100:.2f}%")
    print(f" Margin at EER:  {results['Margin_at_EER']:.6f}")
    print(f" Avg SRR:        {results['Avg_SRR']:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
