
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock
from main import ActionAwareMixin

class MockEngine(ActionAwareMixin):
    def __init__(self):
        self.use_gpu = False
        self.datamanager = MagicMock()
        self.datamanager.width = 128
        self.datamanager.height = 256
        self.datamanager.data_type = 'image'
        self.datamanager.fetch_test_loaders.return_value = {}

    def parse_data_for_eval(self, data):
        # return imgs, pids, camids
        return data['img'], data['pid'], data['camid']

    def extract_features(self, input):
        # return dummy features
        return input

    def export_ranking_results_for_ext_eval(self, *args):
        pass

class TestActionAwareEvaluation(unittest.TestCase):
    def test_action_masking(self):
        engine = MockEngine()
        
        # Create dummy data
        # 3 query images, 3 gallery images
        # Query Actions: [0, 0, 1]
        # Gallery Actions: [0, 1, 0]
        
        # Expected matches (Action ID must match):
        # Q0 (A0) -> G0 (A0) : Match possible
        # Q0 (A0) -> G1 (A1) : MASKED (Inf)
        # Q0 (A0) -> G2 (A0) : Match possible
        
        # Q1 (A0) -> G0 (A0) : Match possible
        # Q1 (A0) -> G1 (A1) : MASKED (Inf)
        # Q1 (A0) -> G2 (A0) : Match possible
        
        # Q2 (A1) -> G0 (A0) : MASKED (Inf)
        # Q2 (A1) -> G1 (A1) : Match possible
        # Q2 (A1) -> G2 (A0) : MASKED (Inf)

        query_loader = [
            {
                'img': torch.randn(3, 10), # 3 images, feature dim 10
                'pid': torch.tensor([1, 2, 3]),
                'camid': torch.tensor([0, 0, 1]) # Action IDs
            }
        ]
        
        gallery_loader = [
             {
                'img': torch.randn(3, 10),
                'pid': torch.tensor([1, 4, 2]),
                'camid': torch.tensor([0, 1, 0]) # Action IDs
            }
        ]
        
        # Mock metrics.compute_distance_matrix to return simple euclidean distance 
        # But here we are testing the Engine's _evaluate which calls metrics.compute_distance_matrix
        # We'll let it use the real metrics function if available or mock it if complex.
        # torchreid should be importable.
        
        # Run _evaluate
        # We need to capture the distmat. 
        # _evaluate returns (rank1, mAP) or None.
        # It prints a lot. We want to check the internal distmat logic.
        # Since _evaluate is a big function, we can inspect it by mocking metrics.evaluate_rank
        # to capture the passed distmat.
        
        with unittest.mock.patch('torchreid.metrics.evaluate_rank') as mock_eval:
            mock_eval.return_value = (np.array([0]), 0.0)
            
            engine._evaluate(
                dataset_name='test_dataset',
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric='euclidean',
                visrank=False,
                rerank=False
            )
            
            # Check args passed to evaluate_rank
            args, _ = mock_eval.call_args
            distmat = args[0]
            
            print("\nCaptured Distmat:\n", distmat)
            
            # Check masking
            # Q0(A0) vs G1(A1) -> index [0, 1] should be huge
            self.assertTrue(distmat[0, 1] > 1e19, "Q0-G1 should be masked")
            
            # Q2(A1) vs G0(A0) -> index [2, 0] should be huge
            self.assertTrue(distmat[2, 0] > 1e19, "Q2-G0 should be masked")
            
            # Q0(A0) vs G0(A0) -> index [0, 0] should be normal (small)
            self.assertTrue(distmat[0, 0] < 1000, "Q0-G0 should NOT be masked")

if __name__ == '__main__':
    unittest.main()
