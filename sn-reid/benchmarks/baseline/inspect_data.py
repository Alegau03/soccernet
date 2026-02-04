
import sys
import os
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), '../../'))

import torch
import torchreid
from default_config import get_default_config, imagedata_kwargs

def inspect_dataset():
    """
    Utility function to inspect the structure of the SoccerNet dataset.
    
    It attempts to:
    1. Initialize the DataManager with default configuration.
    2. Check if the Training Set loads correctly.
    3. Inspect the format of a training sample (verifying keys/structure).
    4. Check if the Test/Query sets load correctly.
    5. Print sample items to verify that paths, PIDs, and Action IDs are present.
    """
    cfg = get_default_config()
    cfg.data.sources = ['soccernetv3']
    cfg.data.targets = ['soccernetv3']
    cfg.use_gpu = True
    # We might need to set root if it's not default
    # cfg.data.root = 'reid-data' 
    
    # Just try to initialize datamanager
    try:
        # Initialize the DataManager which handles downloading, unzipping, and parsing splits.
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
        print("DataManager initialized successfully")
        
        # Check train/test dataset structure
        if datamanager.train_loader:
             print("Train loader exists")
             dataset = datamanager.train_loader.dataset
             if len(dataset) > 0:
                 item = dataset[0]
                 print(f"Train item type: {type(item)}")
                 print(f"Train item: {item}")
                 # Expected format: (image_tensor, pid, camid/action_id, img_path)
                 
        if datamanager.test_loader:
            print("Test loader keys:", datamanager.test_loader.keys())
            for name, loader in datamanager.test_loader.items():
                print(f"Checking {name} loader")
                dataset = loader['query'].dataset
                if len(dataset) > 0:
                    item = dataset[0]
                    print(f"Query item 0: {item}")
                    # check if action is in there. 
                    # item might be a tuple or a dict.
                    # if standard torchreid, likely (img_path, pid, camid) or (img, pid, camid)
                    # if soccernetv3 is custom, maybe it has more.
                    
                    # Also check dataset.data (the raw list of tuples before transforms)
                    # We expect: (image_path, pid, action_idx)
                    raw_item = dataset.dataset[0] 
                    print(f"Raw dataset item 0: {raw_item}")
                    
    except Exception as e:
        print(f"Error initializing datamanager: {e}")

if __name__ == "__main__":
    inspect_dataset()
