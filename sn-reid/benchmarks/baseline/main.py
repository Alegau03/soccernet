import sys
import time
import os
import os.path as osp
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*oneDNN custom operations.*')
warnings.filterwarnings('ignore', message='Cython evaluation.*')

# Add project root to sys.path to allow importing torchreid
sys.path.insert(0, osp.join(osp.dirname(__file__), '../../'))

import argparse
import torch
import torch.nn as nn

import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)


import numpy as np
import time
import datetime
import os.path as osp
import json
from torch.nn import functional as F
from torchreid import metrics
from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results
)

class ActionAwareMixin:
    """
    Mixin class that overrides the default evaluation logic to support Action-Aware Re-identification.
    It introduces specific logic to handle 'action' labels from the SoccerNet dataset, ensuring that
    re-identification only happens between players performing the same action class (e.g., matching
    'dribbling' queries only against 'dribbling' gallery items).
    """
    
    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        eval_metric='default',
        ranks=[1, 5, 10, 20],
        rerank=False,
        export_ranking_results=False
    ):
        """
        Custom evaluation method for Action-Aware ReID.
        
        This method performs the following steps:
        1. Extracts features from the Query set.
        2. Extracts features from the Gallery set.
        3. Computes the distance matrix between Query and Gallery features.
        4. Applies an 'Action Mask': artificially sets the distance to infinity between samples
           that belong to different action classes (using 'camids' as action placeholders).
           This strictly enforces that we only retrieve candidates from the same action category.
        5. Computes standard ReID metrics (mAP, CMC Rank-k).
        """
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            """
            Extracts deep features from a given data loader.
            Returns the feature matrix, list of person IDs (pids), and list of action IDs (camids).
            """
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self.extract_features(imgs)
                # Test-Time Augmentation: Horizontal Flip
                features_flipped = self.extract_features(torch.flip(imgs, [3]))
                features = features + features_flipped
                batch_time.update(time.time() - end)
                features = features.cpu().clone()
                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_

        print('Extracting features from query set ...')
        qf, q_pids, q_camids = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix with metric={} ...'.format(dist_metric)
        )
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        # --- ACTION AWARE MASKING ---
        # The core logic of Action-Aware ReID:
        # We misuse the 'camid' field to store the Action ID (e.g., 'Goal', 'Foul').
        # Here, we create a mask to identify mismatches between the query's action and gallery's action.
        print('Applying Action-Aware Masking...')
        # Create a boolean mask where query action != gallery action
        # Shape: (num_query, num_gallery)
        
        action_mismatch = (q_camids[:, np.newaxis] != g_camids[np.newaxis, :])
        
        if rerank:
            print('Applying person re-ranking (Action-Aware) ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            
            distmat_qq = distmat_qq.numpy()
            distmat_gg = distmat_gg.numpy()
            
            # Pass CLEAN matrices to re-ranking
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)
            
        # Apply mask to the FINAL distance matrix
        # Set mismatching distances to a large number (5000) effectively removing them from consideration.
        # This forces the model to ignore gallery samples that have a different action than the query.
        # 5000 is safe for float32/16 operations and sufficient (squared is 2.5e7)
        distmat[action_mismatch] = 5000

        if export_ranking_results:
            self.export_ranking_results_for_ext_eval(distmat, q_pids, q_camids, g_pids, g_camids, save_dir, dataset_name)

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.fetch_test_loaders(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk,
                display_border=not query_loader.dataset.hidden_labels,
            )

        if not query_loader.dataset.hidden_labels:
            print('Computing CMC and mAP ...')
            cmc, mAP = metrics.evaluate_rank(
                distmat,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                eval_metric=eval_metric
            )
            print('** Results **')
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
            return cmc[0], mAP
        else:
            print("Couldn't compute CMC and mAP because of hidden identity labels.")
            return None, None

class ActionAwareImageSoftmaxEngine(ActionAwareMixin, torchreid.engine.ImageSoftmaxEngine):
    pass

class ActionAwareImageTripletEngine(ActionAwareMixin, torchreid.engine.ImageTripletEngine):
    pass


def build_datamanager(cfg):
    """
    Constructs the DataManager based on configuration.
    It handles data loading, transforms, and batch creation for both Image and Video datasets.
    """
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    """
    Constructs the Training/Evaluation Engine using the custom ActionAware classes.
    The engine manages the training loop, loss calculation (Softmax or Triplet), and evaluation.
    """
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = ActionAwareImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

        else:
            engine = ActionAwareImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    else:
        # Video engines can be extended similarly if needed
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


def main():
    """
    Main entry point of the training script.
    1. Parses arguments and configuration.
    2. Sets up logging and environment.
    3. Builds the DataManager (datasets).
    4. Builds the Model architecture (e.g., OSNet).
    5. Sets up Optimizer and Learning Rate Scheduler.
    6. Loads checkpoints if resuming.
    7. Builds the Engine and starts the training/evaluation loop.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
