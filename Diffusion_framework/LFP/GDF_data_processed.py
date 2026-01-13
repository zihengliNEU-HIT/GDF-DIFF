import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from torchvision import transforms
import os
import re
import pickle


class BatteryCAFDataset(Dataset):
    """
    Battery CAF image dataset for diffusion model training.
    Loads fragment GAF images (FRAG) as conditions and full GAF images as targets,
    with conditioning vectors from MATLAB data files.
    
    Args:
        data_dir: Directory containing .mat data files
        split: Data split ('train', 'val', or 'test')
        transform: Optional data transforms
        cond_mean: Mean for conditioning vector normalization (from training set)
        cond_std: Std for conditioning vector normalization (from training set)
    """
    def __init__(self, data_dir, split='train', transform=None, cond_mean=None, cond_std=None):
        self.data_dir = data_dir
        self.split = split.lower()
        self.transform = transform
        self.data_files = []

        # Search for .mat files based on split
        if self.split == 'train':
            pat = re.compile(r'.*_train_sliding\.mat$', re.I)
        elif self.split == 'val':
            pat = re.compile(r'.*_val_sliding\.mat$', re.I)
        elif self.split == 'test':
            pat = re.compile(r'.*_test_sliding\.mat$', re.I)
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        for fname in os.listdir(data_dir):
            if pat.match(fname):
                self.data_files.append(os.path.join(data_dir, fname))

        if not self.data_files:
            print(f"Warning: No '*_{self.split}_sliding.mat' files found in {data_dir}")

        # Statistics
        self.total_cycles = 0
        self.skipped_empty_segments = 0
        self.valid_segments = 0

        # Flattened sample list
        self.flat_samples = []

        # Load and process data
        for file in self.data_files:
            if os.path.exists(file):
                self._load_and_flatten_data(file)
            else:
                print(f"Warning: {file} does not exist")

        print(f"Loaded {len(self.flat_samples)} valid samples for {split}")
        print(f"Statistics: total_cycles={self.total_cycles}, valid_segments={self.valid_segments}, skipped_empty={self.skipped_empty_segments}")

        # Compute or use provided conditioning vector statistics
        if len(self.flat_samples) > 0:
            if cond_mean is not None and cond_std is not None:
                self.cond_mean = np.asarray(cond_mean, dtype=np.float32)
                self.cond_std = np.asarray(cond_std, dtype=np.float32)
                self.cond_std = np.where(self.cond_std < 1e-8, 1.0, self.cond_std)
                print(f"Conditioning vector normalization [{self.split}]: using training set mean/std")
            else:
                self._compute_cond_stats()

    def _compute_cond_stats(self):
        """
        Compute mean and standard deviation of conditioning vectors for Z-score normalization.
        """
        all_cond_vecs = [sample['cond_vec'] for sample in self.flat_samples]
        all_cond_vecs = np.stack(all_cond_vecs, axis=0)
        
        self.cond_mean = np.mean(all_cond_vecs, axis=0)
        self.cond_std = np.std(all_cond_vecs, axis=0)
        
        # Avoid division by zero
        self.cond_std = np.where(self.cond_std < 1e-8, 1.0, self.cond_std)
        
        print(f"Conditioning vector normalization [{self.split}]: mean={self.cond_mean}, std={self.cond_std}")

    def _to_1d_float_vec(self, x):
        """
        Convert MATLAB cell content to 1D float32 vector.
        Handles scalars, (1,d), (d,1), and nested cell arrays.
        
        Args:
            x: Input data from MATLAB cell
        Returns:
            1D float32 numpy array
        """
        arr = np.array(x, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = np.array([arr], dtype=np.float32)
        return arr.astype(np.float32)

    def _load_and_flatten_data(self, file_path):
        """
        Load .mat file and flatten into individual samples.
        Each sample contains fragment image, conditioning vector, and target full image.
        
        Args:
            file_path: Path to .mat file
        """
        data = loadmat(file_path)

        # battery_type ⚪
        file_name = os.path.basename(file_path)
        if '53C_54' in file_name:
            battery_type = '53C_54'
        elif '56C_19' in file_name:
            battery_type = '56C_19'
        elif '56C_36' in file_name:
            battery_type = '56C_36'
        else:
            battery_type = 'unknown'

        # Required fields
        gaf_cell = data.get('GAF_cell', None)
        frag_cell = data.get('FRAG_cell', None)
        cond_cell = data.get('COND_VEC_cell', None)
        info_cell = data.get('info_cell', None)
        seg_idx_cell = data.get('SEG_idx_cell', None)

        # Check required fields
        if any(x is None for x in [gaf_cell, frag_cell, cond_cell, info_cell]):
            print(f"Warning: {file_path} missing GAF/FRAG/COND_VEC/info fields, skipping")
            return

        n_samples = gaf_cell.shape[0]
        n_segments = frag_cell.shape[1]
        self.total_cycles += n_samples

        for i in range(n_samples):
            full_caf = gaf_cell[i, 0]
            if full_caf.size == 0:
                continue

            # Parse cycle info as [battery_id, cycle_idx]
            cycle_info = info_cell[i, 0]
            if isinstance(cycle_info, np.ndarray):
                if cycle_info.ndim == 2 and cycle_info.shape == (1, 2):
                    battery_id = int(cycle_info[0, 0])
                    cycle_idx = int(cycle_info[0, 1])
                elif cycle_info.ndim == 1 and cycle_info.size >= 2:
                    battery_id = int(cycle_info[0])
                    cycle_idx = int(cycle_info[1])
                else:
                    flat_info = cycle_info.flatten()
                    battery_id = int(flat_info[0])
                    cycle_idx = int(flat_info[1])
                cycle_info = [battery_id, cycle_idx]
            elif isinstance(cycle_info, (list, tuple)):
                cycle_info = [int(cycle_info[0]), int(cycle_info[1])]
            else:
                print(f"Warning: Unknown info format: {type(cycle_info)}")
                cycle_info = [0, i+1]

            for s in range(n_segments):
                frag = frag_cell[i, s]
                cond = cond_cell[i, s]

                if seg_idx_cell is not None and seg_idx_cell[i, s].size > 0:
                    segment_idx = int(seg_idx_cell[i, s][0, 0])
                else:
                    segment_idx = s

                # Skip empty fragments
                if frag.size == 0:
                    self.skipped_empty_segments += 1
                    continue
                
                # Skip if conditioning vector is missing
                if cond is None or (isinstance(cond, np.ndarray) and cond.size == 0):
                    self.skipped_empty_segments += 1
                    continue

                # Store sample
                self.flat_samples.append({
                    'full_caf': full_caf,
                    'frag': frag,
                    'cond_vec': self._to_1d_float_vec(cond),
                    'info': cycle_info,
                    'battery_type': battery_type,  # battery type⚪
                    'segment_idx': segment_idx
                })
                self.valid_segments += 1

    def __len__(self):
        return len(self.flat_samples)

    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        Returns:
            Dictionary containing:
                - input: Fragment image [1, H, W] in range [-1, 1]
                - target: Full CAF image [1, H, W] in range [-1, 1]
                - cond_vec: Normalized conditioning vector [d]
                - info: Battery and cycle information
                - segment_idx: Segment index
        """
        sample = self.flat_samples[idx]
        full_caf = sample['full_caf']
        frag = sample['frag']
        cond_vec = sample['cond_vec']

        # Normalize images to [0, 1]
        full_caf = full_caf.astype(np.float32) / 255
        frag = frag.astype(np.float32) / 255

        # Add channel dimension
        input_tensor = frag[np.newaxis, :, :]
        target_tensor = full_caf[np.newaxis, :, :]

        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        # Map images to [-1, 1]
        input_tensor = torch.from_numpy(input_tensor).float() * 2.0 - 1.0
        target_tensor = torch.from_numpy(target_tensor).float() * 2.0 - 1.0
        
        # Z-score normalize conditioning vector
        cond_vec = (cond_vec - self.cond_mean) / self.cond_std
        cond_vec = torch.from_numpy(cond_vec).float()

        return {
            'input': input_tensor,
            'target': target_tensor,
            'cond_vec': cond_vec,
            'info': sample['info'],
            'battery_type': sample['battery_type'],  # battery type ⚪
            'segment_idx': sample['segment_idx']
        }


def custom_collate_fn(batch):
    """
    Custom collate function for batching samples.
    
    Args:
        batch: List of samples from __getitem__
    Returns:
        Dictionary of batched tensors
    """
    inputs = torch.stack([item['input'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    cond_vecs = torch.stack([item['cond_vec'] for item in batch])
    infos = [item['info'] for item in batch]
    battery_types = [item['battery_type'] for item in batch]   # battery type ⚪
    segment_idxs = torch.tensor([item['segment_idx'] for item in batch])
    
    return {
        'input': inputs,
        'target': targets,
        'cond_vec': cond_vecs,
        'info': infos,
        'battery_type': battery_types,  # battery type ⚪
        'segment_idx': segment_idxs
    }


def save_normalization_stats(data_dir, cond_mean, cond_std):
    """
    Save normalization statistics to pickle file.
    
    Args:
        data_dir: Directory to save statistics file
        cond_mean: Mean values for normalization
        cond_std: Standard deviation values for normalization
    """
    stats = {
        'cond_mean': cond_mean,
        'cond_std': cond_std
    }
    
    stats_path = os.path.join(data_dir, 'normalization_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"Normalization statistics saved to {stats_path}")
    print(f"  Mean shape: {cond_mean.shape}")
    print(f"  Std shape: {cond_std.shape}")


def load_normalization_stats(data_dir):
    """
    Load normalization statistics from pickle file.
    
    Args:
        data_dir: Directory containing statistics file
    Returns:
        Tuple of (cond_mean, cond_std)
    """
    stats_path = os.path.join(data_dir, 'normalization_stats.pkl')
    
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Normalization statistics file not found: {stats_path}\n"
        )
    
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    cond_mean = stats['cond_mean']
    cond_std = stats['cond_std']
    
    print(f"Normalization statistics loaded from {stats_path}")
    print(f"  Mean shape: {cond_mean.shape}")
    print(f"  Std shape: {cond_std.shape}")
    
    return cond_mean, cond_std


def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    """
    Create data loaders for training, validation, and testing.
    Automatically saves normalization statistics when training dataset is created.
    
    Args:
        data_dir: Directory containing .mat data files
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    pin = torch.cuda.is_available()
    
    # Create training dataset first to extract normalization statistics
    train_dataset = BatteryCAFDataset(data_dir, split='train', transform=None)
    cm, cs = train_dataset.cond_mean, train_dataset.cond_std

    # Save normalization statistics for future use
    save_normalization_stats(data_dir, cm, cs)

    # Create validation and test datasets using training set statistics
    val_dataset = BatteryCAFDataset(data_dir, split='val', transform=None, cond_mean=cm, cond_std=cs)
    test_dataset = BatteryCAFDataset(data_dir, split='test', transform=None, cond_mean=cm, cond_std=cs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=custom_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=custom_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=custom_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader

def get_reverse_dataloader(data_dir, batch_size=16, num_workers=4):
    """
    Create reverse diffusion data loader using pre-saved normalization statistics.
    Does not require training data to be present.
    
    Args:
        data_dir: Directory containing .mat test files and normalization_stats.pkl
        batch_size: Batch size for data loader
        num_workers: Number of worker processes for data loading
    Returns:
        Test data loader for reverse diffusion
    """
    pin = torch.cuda.is_available()
    
    # Load pre-saved normalization statistics
    cond_mean, cond_std = load_normalization_stats(data_dir)
    
    # Create test dataset using loaded statistics
    test_dataset = BatteryCAFDataset(
        data_dir, 
        split='test', 
        transform=None, 
        cond_mean=cond_mean, 
        cond_std=cond_std
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=custom_collate_fn,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return test_loader