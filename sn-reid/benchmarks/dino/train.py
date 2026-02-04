import sys
import os
import time
import argparse
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='Cython evaluation.*')
warnings.filterwarnings('ignore', message='To support symlinks on Windows.*')

import torch
from torch import nn
from torch.nn import functional as F

# Add project root to sys.path to allow importing torchreid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

# --- CHECK DEPENDENCIES ---
try:
    import torchreid
    from torchreid.utils import (
        Logger, check_isfile, set_random_seed, collect_env_info,
        resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
    )
except ImportError:
    print("Error: 'torchreid' library not found. Please install it or run within the SoccerNet baseline environment.")
    sys.exit(1)

try:
    from transformers import Dinov2Model
    from peft import LoraConfig, get_peft_model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Error: 'transformers' or 'peft' not found. Please install: pip install transformers peft")
    sys.exit(1)

# --- CUSTOM MODEL DEFINITION (Embedded for Portability) ---
# This ensures the partner doesn't need to modify their torchreid/models/__init__.py
class DINOv2ReID(nn.Module):
    def __init__(self, num_classes, loss='softmax', pretrained=True, use_gpu=True, model_name='facebook/dinov2-small', use_lora=True):
        super(DINOv2ReID, self).__init__()
        self.loss = loss
        self.use_gpu = use_gpu
        self.in_planes = 768 if 'base' in model_name else 384
        
        # Load Backbone
        print(f"Loading DINOv2 model: {model_name}")
        self.backbone = Dinov2Model.from_pretrained(model_name)
        
        if use_lora:
            print("Applying LoRA to DINOv2 backbone...")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["query", "value", "key", "dense"], 
                lora_dropout=0.1,
                bias="none",
                modules_to_save=[],
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()

        # Classifier Head
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        
        # OCR Head (Optional but included for compatibility)
        self.num_ocr_classes = 101
        self.classifier_ocr = nn.Linear(self.in_planes, self.num_ocr_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.classifier_ocr.apply(weights_init_classifier)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0, :] # [CLS] token
        
        v = features
        if self.bottleneck:
            v_bn = self.bottleneck(v)
        else:
            v_bn = v

        if not self.training:
            return v_bn

        y = self.classifier(v_bn)
        y_ocr = self.classifier_ocr(v_bn)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            # return y, y_ocr, v_bn # Original (caused crash)
            return y, v_bn
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

# --- MAIN TRAINING SCRIPT ---
# --- MAIN TRAINING SCRIPT ---
def main():
    parser = argparse.ArgumentParser(description='Standalone DINOv2 Fine-tuning')
    parser.add_argument('--root', type=str, default='data', help='Path to SoccerNet dataset root')
    parser.add_argument('--weights', type=str, default='', help='Path to pretrained weights (e.g., from 30% run)')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--save-dir', type=str, default='log/dinov2_standalone', help='Directory to save logs and models')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=60, help='Max epochs')
    parser.add_argument('--eval-freq', type=int, default=1, help='Evaluation frequency (epochs)')
    parser.add_argument('--eval-metric', type=str, default='soccernetv3', help='Evaluation metric: "default" (Market1501) or "soccernetv3"')
    parser.add_argument('--print-freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--fp16', action='store_true', help='Use Mixed Precision Training (Auto-enabled if GPU available)')
    parser.add_argument('--gpu-id', type=str, default='0', help='GPU ID')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_gpu = torch.cuda.is_available()
    
    # --- RTX 3060 OPTIMIZATION (Ampere) ---
    if use_gpu:
        print("Enabling TF32 (TensorFloat-32) for RTX 30xx optimization...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 1. Logger setup
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    sys.stdout = Logger(os.path.join(args.save_dir, 'train.log'))
    print(f"Saving logs to {args.save_dir}")

    # 2. Data Manager
    print("Initializing Data Manager...")
    datamanager = torchreid.data.ImageDataManager(
        root=args.root,
        sources='soccernetv3',
        targets='soccernetv3',
        height=384,
        width=192,
        batch_size_train=args.batch_size,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop', 'random_erase'],
        train_sampler='RandomIdentitySampler',
        num_instances=4,
        workers=4, # Optimized for 12GB VRAM / IO
        soccernetv3_training_subset=1.0 # FULL DATASET
    )

    # 3. Model Build
    print("Building DINOv2 Model (Small + LoRA)...")
    # We instantiate our local class directly
    model = DINOv2ReID(
        num_classes=datamanager.num_train_pids,
        loss='triplet',
        pretrained=True,
        use_gpu=use_gpu,
        model_name='facebook/dinov2-small',
        use_lora=True
    )
    
    if use_gpu:
        model = model.cuda()
    
    # Load pretrained weights if provided (AND NOT resuming)
    if not args.resume and args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        torchreid.utils.load_pretrained_weights(model, args.weights)
    elif args.weights and not args.resume:
         print(f"WARNING: Weights file {args.weights} not found! Starting from scratch (ImageNet).")

    # 4. Optimizer & Scheduler
    print(f"Setting up Optimizer (AdamW, lr={args.lr})...")
    # Only optimize open layers (LoRA + Classifier)
    open_layers = ['backbone', 'classifier', 'classifier_ocr', 'bottleneck'] 
    
    params = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            params += [{"params": [value], "lr": args.lr, "weight_decay": 5e-4}]
            
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=5e-4)
    
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='cosine',
        max_epoch=args.epochs
    )

    # --- RESUME LOGIC ---
    start_epoch = 0
    if args.resume and check_isfile(args.resume):
        start_epoch = resume_from_checkpoint(
            args.resume, model, optimizer=optimizer, scheduler=scheduler
        )
        print(f"=> Resuming from epoch {start_epoch}")
        
        # NOTE: With Cosine Scheduler, we do NOT force the LR manually.
        # The scheduler will automatically adjust it based on the start_epoch.

        

        # Rebuild Scheduler to match new start epoch (optional but recommended)
        # scheduler = torchreid.optim.build_lr_scheduler(optimizer, lr_scheduler='multi_step', stepsize=[10])

        # COPY BEST MODEL for continuity
        # engine.run looks for 'model-best.pth.tar' in save_dir to initialize best_rank1
        resume_dir = os.path.dirname(args.resume)
        prev_best_path = os.path.join(resume_dir, 'model-best.pth.tar')
        new_best_path = os.path.join(args.save_dir, 'model-best.pth.tar')
        if os.path.exists(prev_best_path):
             import shutil
             print(f"=> Copying previous best model from {prev_best_path} to {new_best_path}")
             shutil.copy(prev_best_path, new_best_path)

    # 5. Engine
    print("Building Engine...")
    # NOTE: 'fp16' argument removed as it's not supported by base ImageTripletEngine 
    # (it uses self.scaler = torch.cuda.amp.GradScaler(enabled=use_gpu) internally)
    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        margin=0.3,
        weight_t=1.0,
        weight_x=1.0, # Cross Entropy
        scheduler=scheduler,
        use_gpu=use_gpu,
        label_smooth=True
    )

    # 6. Run
    print("Starting Training...")
    engine.run(
        save_dir=args.save_dir,
        max_epoch=args.epochs,
        start_epoch=start_epoch, 
        eval_freq=args.eval_freq,
        print_freq=args.print_freq,
        test_only=False,
        dist_metric='euclidean',
        eval_metric=args.eval_metric,
        normalize_feature=True,
        visrank=False
    )

if __name__ == '__main__':
    main()
