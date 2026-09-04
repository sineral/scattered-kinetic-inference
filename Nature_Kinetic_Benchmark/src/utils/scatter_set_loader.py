"""
PyTorch Dataset and DataLoader for Nature dataset to Set Transformer format.

This module provides efficient data loading and on-the-fly generation of set-based
training data from kinetic profile data. Each set contains randomly sampled points
from concentration profiles, formatted for Set Transformer training.

Generalized to handle:
- Variable number of concentration profiles (auto-detected from data shape)
- Variable temporal length (auto-detected from data shape)
- x2 structure: [time, substrate, product] repeated for each profile

Data Format:
- Input: 
  - x1: catalyst concentrations (n_samples, n_profiles)
  - x2: kinetic profiles (n_samples, n_timepoints, n_profiles*3)
  - y: labels (n_samples, 1)
- Output: Sets of points with 6 basic features plus optional engineered features
  (up to 20-D: catalyst loading, initial substrate/product, reaction time,
  residual substrate, generated product, and rates/ratios).
- Each set contains n_points randomly sampled from [n_min, n_max]
- Points are evenly distributed across all concentration profiles
- Temporal features (rates, fractions) help model learn reaction dynamics
"""

import os
import pickle
import yaml
import numpy as np
import gc
from typing import Dict, List, Tuple, Union, Optional

import torch
from torch.utils.data import Dataset, DataLoader

# Names follow the paper table. Formulas are the Nature ST checkpoint construction.
FEATURE_NAMES = [
    "Catalyst loading",       # 0  [cat]_0
    "Initial substrate",      # 1  [S]_0
    "Initial product",        # 2  [P]_0
    "Reaction time",          # 3  t
    "Residual substrate",     # 4  [S]_t
    "Generated product",      # 5  [P]_t
    "Substrate rate",         # 6  ([S]_0 - [S]_t) / (t + eps)
    "Product rate",           # 7  ([P]_t - [P]_0) / (t + eps)
    "Substrate fraction",     # 8  [S]_t / ([S]_0 + eps)
    "Conversion",             # 9  [P]_t / ([S]_0 + [P]_0 + eps)
    "Total material",         # 10 [S]_t + [P]_t
    "Mass deviation",         # 11 |([S]_t + [P]_t) - ([S]_0 + [P]_0)| / ([S]_0 + [P]_0 + eps)
    "Product fraction",       # 12 ([S]_0 - [S]_t) / ([S]_0 + eps)
    "Substrate ratio",        # 13 ln(max(([S]_t + eps) / ([S]_0 + eps), eps))
    "Product ratio",          # 14 ln(max(([P]_t + eps) / ([P]_0 + eps), eps))
    "Specific productivity",  # 15 ([P]_t - [P]_0) / ([cat]_0 * t + eps)
    "Substrate/catalyst",     # 16 [S]_t / ([cat]_0 + eps)
    "Product/catalyst",       # 17 [P]_t / ([cat]_0 + eps)
    "Substrate/product",      # 18 [S]_t / ([P]_t + eps)
    "Turnover number",        # 19 ([P]_t - [P]_0) / ([cat]_0 + eps)
]


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def mechanism_name_to_id(mechanism_names: List[str]) -> List[int]:
    """
    Convert mechanism names (e.g., 'M1', 'M4') to IDs (e.g., 0, 3).
    
    Args:
        mechanism_names: List of mechanism names like ['M1', 'M4', 'M10']
        
    Returns:
        List of mechanism IDs (0-indexed)
    """
    mechanism_ids = []
    for name in mechanism_names:
        # Remove 'M' prefix and convert to 0-indexed
        mech_id = int(name.replace('M', '')) - 1
        mechanism_ids.append(mech_id)
    return mechanism_ids


def filter_by_mechanism(x: np.ndarray, y: np.ndarray, mechanism_ids: List[int]) -> tuple:
    """
    Filter dataset by mechanism types.
    
    Args:
        x: Feature data (can be x1 or x2)
        y: Labels
        mechanism_ids: List of mechanism IDs to keep
        
    Returns:
        Tuple of (filtered_x, filtered_y)
    """
    # Create mask for selected mechanisms
    mask = np.isin(y.flatten(), mechanism_ids)
    
    # Apply filter
    filtered_x = x[mask]
    filtered_y = y[mask]
    
    return filtered_x, filtered_y


def filter_test_data(x2_test: Dict, y_test: np.ndarray, mechanism_ids: List[int]) -> Dict:
    """
    Filter test data dictionary by mechanism types.
    
    Args:
        x2_test: Test kinetic profiles dictionary with structure [timepoints][error_level]
        y_test: Test labels
        mechanism_ids: List of mechanism IDs to keep
        
    Returns:
        Filtered x2_test dictionary with same structure
    """
    # Create mask for filtering
    mask = np.isin(y_test.flatten(), mechanism_ids)
    
    # Filter the nested dictionary structure
    filtered_x2_test = {}
    for timepoints in x2_test:
        filtered_x2_test[timepoints] = {}
        for error_level in x2_test[timepoints]:
            filtered_x2_test[timepoints][error_level] = x2_test[timepoints][error_level][mask]
    
    return filtered_x2_test


class NatureSetDataset(Dataset):
    """
    PyTorch Dataset for generating set-based samples from Nature kinetic profiles.
    
    This dataset loads all data into memory and generates training samples on-the-fly
    by randomly sampling points from concentration profiles. Each sample is a set of
    points with format [catalyst loading, initial substrate, initial product,
    reaction time, residual substrate, generated product].
    
    Generalized to handle:
    - Variable number of concentration profiles (detected from x2 shape)
    - Variable temporal length (detected from x2 shape)
    - x2 structure: [time, substrate, product] repeated for each profile
    
    Args:
        x1: Initial catalyst concentrations (n_samples, n_profiles)
        x2: Kinetic profiles (n_samples, n_timepoints, n_profiles*3) or dict for Nature test data
            - Standard format: np.ndarray with shape (n_samples, n_timepoints, n_profiles*3)
            - Nature test format: Dict[timepoints][error_level] -> np.ndarray
            - Each profile has 3 columns: [time, substrate, product]
            - Total columns = n_profiles * 3
        y: Labels (n_samples, 1)
        n_min: Minimum number of points per set
        n_max: Maximum number of points per set
        samples_per_epoch: Number of samples to generate per epoch (for training)
        is_nature_test: Whether this is Nature paper test data with nested dict structure
                        (standard array-format splits should use False)
        nature_test_timepoints: Number of timepoints for Nature test data (2, 6, or 20)
        nature_test_error: Error level for Nature test data (0, 1, or 5)
        noise_enabled: Whether to add Gaussian noise to end concentrations
        noise_std_dev: Relative error std dev (e.g., 0.01 = 1% error) or list of errors
    """
    
    def __init__(
        self,
        x1: np.ndarray,
        x2: Union[np.ndarray, Dict],
        y: np.ndarray,
        n_min: int,
        n_max: int,
        samples_per_epoch: Optional[int] = None,
        is_nature_test: bool = False,
        nature_test_timepoints: int = 20,
        nature_test_error: int = 1,
        noise_enabled: bool = False,
        noise_std_dev: Union[float, List[float]] = 0.01,
        feature_config: Optional[Dict] = None
    ):
        self.x1 = x1
        y_flat = y.flatten()  # Shape: (n_samples,)
        
        # Store noise parameters
        self.noise_enabled = noise_enabled
        self.noise_std_dev = noise_std_dev
        
        # Store feature configuration
        if feature_config is None:
            # Default: 6 basic features; optional engineered features may be added.
            self.feature_config = {
                'basic': [
                    'catalyst_loading', 'initial_substrate', 'initial_product',
                    'reaction_time', 'residual_substrate', 'generated_product',
                ],
                'optional_features': {
                    'enabled': True,
                    'features': ['substrate_rate', 'product_rate', 'substrate_fraction', 'conversion']
                }
            }
        else:
            self.feature_config = feature_config
        
        # Calculate feature dimension
        self.feature_dim = len(self.feature_config.get('basic', []))
        if self.feature_config.get('optional_features', {}).get('enabled', False):
            self.feature_dim += len(self.feature_config['optional_features'].get('features', []))
        
        # Create label mapping: original labels -> [0, 1, 2, ...]
        unique_labels = np.unique(y_flat)
        self.label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.inverse_label_map = {new_label: old_label for old_label, new_label in self.label_map.items()}
        
        # Remap labels to consecutive integers starting from 0
        self.y = np.array([self.label_map[label] for label in y_flat])
        
        self.n_min = n_min
        self.n_max = n_max
        self.is_nature_test = is_nature_test
        
        # Nature test x2 is a nested dict; train/val x2 is a dense array.
        if is_nature_test and isinstance(x2, dict):
            self.x2 = x2[nature_test_timepoints][nature_test_error]
        else:
            self.x2 = x2
        
        # For training, we can generate more samples per epoch than actual data
        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
        else:
            self.samples_per_epoch = len(self.x1)
        
        # Get number of actual samples and timepoints
        self.n_samples = len(self.x1)
        self.n_timepoints = self.x2.shape[1]  # Temporal length (variable)
        
        # Dynamically detect number of profiles from x2 shape
        # x2 has shape (n_samples, n_timepoints, n_columns)
        # Each profile has 3 columns: [time, substrate, product]
        n_columns = self.x2.shape[2]
        self.n_profiles = n_columns // 3
        
        # Validate that x1 and x2 have consistent profile counts
        assert self.x1.shape[1] == self.n_profiles, \
            f"Mismatch: x1 has {self.x1.shape[1]} profiles, but x2 implies {self.n_profiles} profiles"
        
        print(f"Initialized NatureSetDataset:")
        print(f"  - Samples: {self.n_samples}")
        print(f"  - Number of profiles: {self.n_profiles}")
        print(f"  - Timepoints per profile: {self.n_timepoints}")
        print(f"  - Samples per epoch: {self.samples_per_epoch}")
        print(f"  - Set size range: [{self.n_min}, {self.n_max}]")
        
        # Build feature dimension description
        feat_desc_parts = [f"basic: {len(self.feature_config.get('basic', []))}"]
        if self.feature_config.get('optional_features', {}).get('enabled', False):
            feat_desc_parts.append(f"optional: {len(self.feature_config['optional_features'].get('features', []))}")
        feat_desc = ", ".join(feat_desc_parts)
        
        print(f"  - Feature dimension: {self.feature_dim} ({feat_desc})")
        
        if self.noise_enabled:
            if isinstance(self.noise_std_dev, (list, tuple, np.ndarray)):
                noise_str = "[" + ", ".join([f"{val:.1%}" for val in self.noise_std_dev]) + "]"
            else:
                noise_str = f"{self.noise_std_dev:.1%}"
            print(f"  - Gaussian noise: enabled (relative error std_dev={noise_str})")
        else:
            print("  - Gaussian noise: disabled")
            
        print(f"  - Label mapping: {self.inverse_label_map}")
    
    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self.samples_per_epoch
    
    def _generate_set_from_sample(self, sample_idx: int) -> Tuple[np.ndarray, int]:
        """
        Generate a set of points from a single kinetic profile sample.
        
        Dynamically handles variable number of profiles and timepoints.
        
        Args:
            sample_idx: Index of the sample to use
            
        Returns:
            Tuple of (set_features, label) where:
            - set_features: (n_points, 10) array with features including temporal dynamics
            - label: Mechanism label
        """
        # Get data for this sample
        cat_init = self.x1[sample_idx]  # Shape: (n_profiles,)
        profile = self.x2[sample_idx]   # Shape: (n_timepoints, n_profiles*3)
        label = self.y[sample_idx]      # Scalar
        
        # Randomly determine number of points for this set
        n_points = np.random.randint(self.n_min, self.n_max + 1)
        
        # Dynamically select noise level for this sample
        current_noise_std = self.noise_std_dev
        if isinstance(current_noise_std, (list, tuple, np.ndarray)):
            current_noise_std = np.random.choice(current_noise_std)
        
        # Allocate array for set features (dimension based on config)
        set_features = np.zeros((n_points, self.feature_dim), dtype=np.float32)
        
        # Distribute points evenly across profiles
        points_per_profile = n_points // self.n_profiles
        remainder = n_points % self.n_profiles
        
        # Randomly select which profiles get the extra points
        profiles_with_extra = np.random.choice(
            self.n_profiles, 
            size=remainder, 
            replace=False
        ) if remainder > 0 else []
        
        point_idx = 0
        for profile_id in range(self.n_profiles):
            # Number of points for this profile
            n_profile_points = points_per_profile + (1 if profile_id in profiles_with_extra else 0)
            
            if n_profile_points == 0:
                continue
            
            # Column indices for this profile in x2
            # Profile 0: cols 0-2, Profile 1: cols 3-5, etc.
            col_offset = profile_id * 3
            
            # Get time, substrate, product columns for this profile
            time_col = profile[:, col_offset]      # Time
            sub_col = profile[:, col_offset + 1]   # Substrate
            prod_col = profile[:, col_offset + 2]  # Product
            
            # Initial values (at time=0, which is index 0)
            sub_0 = sub_col[0]
            prod_0 = prod_col[0]
            cat_0 = cat_init[profile_id]
            
            # Randomly sample timepoint indices (excluding t=0)
            available_indices = np.arange(1, self.n_timepoints)
            sampled_indices = np.random.choice(
                available_indices,
                size=n_profile_points,
                replace=True  # Allow repetition
            )
            
            # Generate features for each sampled point
            for i, time_idx in enumerate(sampled_indices):
                time_t = time_col[time_idx]
                sub_t = sub_col[time_idx]
                prod_t = prod_col[time_idx]
                
                # Apply Gaussian noise to end concentrations if enabled
                # Noise is relative to the concentration value (proportional error)
                if self.noise_enabled:
                    sub_t_noise = sub_t * np.random.normal(0, current_noise_std)
                    prod_t_noise = prod_t * np.random.normal(0, current_noise_std)
                    sub_t = sub_t + sub_t_noise
                    prod_t = prod_t + prod_t_noise
                    # Ensure concentrations remain non-negative
                    sub_t = max(0.0, sub_t)
                    prod_t = max(0.0, prod_t)
                
                # Build features based on configuration (Nature ST formulas)
                features = []

                feature_map = {
                    'catalyst_loading': cat_0,
                    'initial_substrate': sub_0,
                    'initial_product': prod_0,
                    'reaction_time': time_t,
                    'residual_substrate': sub_t,
                    'generated_product': prod_t,
                }

                for feat_name in self.feature_config.get('basic', []):
                    features.append(feature_map.get(feat_name, 0.0))

                if self.feature_config.get('optional_features', {}).get('enabled', False):
                    optional_feature_map = {
                        'substrate_rate': (sub_0 - sub_t) / (time_t + 1e-8),
                        'product_rate': (prod_t - prod_0) / (time_t + 1e-8),
                        'substrate_fraction': sub_t / (sub_0 + 1e-8),
                        'conversion': prod_t / (sub_0 + prod_0 + 1e-8),
                        'total_material': sub_t + prod_t,
                        'mass_deviation': abs((sub_t + prod_t) - (sub_0 + prod_0)) / (sub_0 + prod_0 + 1e-8),
                        'product_fraction': (sub_0 - sub_t) / (sub_0 + 1e-8),
                        'substrate_ratio': np.log(np.maximum((sub_t + 1e-8) / (sub_0 + 1e-8), 1e-8)),
                        'product_ratio': np.log(np.maximum((prod_t + 1e-8) / (prod_0 + 1e-8), 1e-8)),
                        'specific_productivity': (prod_t - prod_0) / (cat_0 * time_t + 1e-8),
                        'substrate_catalyst': sub_t / (cat_0 + 1e-8),
                        'product_catalyst': prod_t / (cat_0 + 1e-8),
                        'substrate_product': sub_t / (prod_t + 1e-8),
                        'turnover_number': (prod_t - prod_0) / (cat_0 + 1e-8),
                    }
                    for feat_name in self.feature_config['optional_features'].get('features', []):
                        features.append(optional_feature_map.get(feat_name, 0.0))
                
                set_features[point_idx] = features
                point_idx += 1
        
        return set_features, label
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get a randomly generated set sample.
        
        Args:
            idx: Index (not used, kept for Dataset interface compatibility)
            
        Returns:
            Tuple of (set_features, label)
        """
        # Always randomly select a sample from the dataset
        # This ensures diversity in both training and validation/test
        sample_idx = np.random.randint(0, self.n_samples)
        
        return self._generate_set_from_sample(sample_idx)


def scatter_set_collate_fn(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for batching variable-length sets.
    
    Args:
        batch: List of (set_features, label) tuples
        
    Returns:
        Tuple of (features, mask, labels) where:
        - features: (batch_size, max_set_size, 6) padded tensor
        - mask: (batch_size, max_set_size) boolean mask (True for real points)
        - labels: (batch_size,) tensor of labels
    """
    # Separate features and labels
    sets = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Get batch size and maximum set size
    batch_size = len(sets)
    max_set_size = max(len(s) for s in sets)
    feature_dim = sets[0].shape[1]  
    
    # Create padded tensor and mask
    features = torch.zeros(batch_size, max_set_size, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_set_size, dtype=torch.bool)
    
    # Fill in the data
    for i, set_data in enumerate(sets):
        set_length = len(set_data)
        features[i, :set_length] = torch.from_numpy(set_data)
        mask[i, :set_length] = True
    
    # Convert labels to tensor
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return features, mask, labels_tensor


def create_dataloader(
    data_dir: str,
    mechanisms: List[str],
    split: str,
    n_min: int,
    n_max: int,
    batch_size: int,
    num_workers: int = 0,
    samples_per_epoch: Optional[int] = None,
    is_nature_test: bool = False,
    nature_test_timepoints: int = 20,
    nature_test_error: int = 1,
    noise_enabled: bool = False,
    noise_std_dev: Union[float, List[float]] = 0.01,
    shuffle: bool = None,
    feature_config: Optional[Dict] = None
) -> Tuple[DataLoader, Dict]:
    """
    Create a single PyTorch DataLoader for a specific data split.
    
    This function loads data from disk, filters by mechanisms, and creates a dataloader
    for one split (train/val/test) at a time, making it easy to switch between 
    different datasets and use in different contexts.
    
    Args:
        data_dir: Directory containing dataset files (e.g., 'data/nature_data')
        mechanisms: List of mechanism names to keep (e.g., ['M1', 'M4', 'M10'])
        split: Data split name ('train', 'val', or 'test')
        n_min: Minimum number of points per set
        n_max: Maximum number of points per set
        batch_size: Batch size for DataLoader
        num_workers: Number of parallel workers for data loading
        samples_per_epoch: Number of samples per epoch (if None, uses dataset size)
        is_nature_test: Whether this is Nature paper test data with nested dict structure
        nature_test_timepoints: Number of timepoints for Nature test data (2, 6, or 20)
        nature_test_error: Error level for Nature test data (0, 1, or 5)
        noise_enabled: Whether to add Gaussian noise to end concentrations
        noise_std_dev: Relative error std dev (e.g., 0.01 = 1% error) or list of errors
        shuffle: Whether to shuffle data (if None, auto-set: True for train, False for val/test)
        
    Returns:
        Tuple of (dataloader, metadata) where metadata contains label_mapping, etc.
    """
    # Auto-determine shuffle if not specified
    if shuffle is None:
        shuffle = (split == 'train')
    
    print(f"Creating {split} dataloader:")
    print(f"  - Data directory: {data_dir}")
    print(f"  - Mechanisms: {mechanisms}")
    print(f"  - Set size range: [{n_min}, {n_max}]")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num workers: {num_workers}")
    print(f"  - Shuffle: {shuffle}")
    if split == 'train' and samples_per_epoch:
        print(f"  - Samples per epoch: {samples_per_epoch}")
    if split == 'test' and is_nature_test:
        print(f"  - Nature test mode: timepoints={nature_test_timepoints}, error={nature_test_error}%")
    if noise_enabled:
        if isinstance(noise_std_dev, (list, tuple, np.ndarray)):
            noise_str = "[" + ", ".join([f"{val:.1%}" for val in noise_std_dev]) + "]"
        else:
            noise_str = f"{noise_std_dev:.1%}"
        print(f"  - Gaussian noise: enabled (relative error std_dev={noise_str})")
    
    # Load and filter data for this split
    print(f"  - Loading {split} data...")
    mechanism_ids = mechanism_name_to_id(mechanisms)
    
    # Find files matching the pattern (ignoring suffix after split name)
    x1_files = [f for f in os.listdir(data_dir) if f.startswith(f'x1_{split}_') and f.endswith('.pkl')]
    y_files = [f for f in os.listdir(data_dir) if f.startswith(f'y_{split}_') and f.endswith('.pkl')]
    x2_files = [f for f in os.listdir(data_dir) if f.startswith(f'x2_{split}_') and f.endswith('.pkl')]
    
    if not x1_files or not y_files or not x2_files:
        raise FileNotFoundError(f"Could not find x1_{split}_*.pkl, x2_{split}_*.pkl, or y_{split}_*.pkl in {data_dir}")
    
    # Load data
    x1_path = os.path.join(data_dir, x1_files[0])
    y_path = os.path.join(data_dir, y_files[0])
    x2_path = os.path.join(data_dir, x2_files[0])
    
    with open(x1_path, 'rb') as f:
        x1_full = pickle.load(f)
    with open(y_path, 'rb') as f:
        y_full = pickle.load(f)
    with open(x2_path, 'rb') as f:
        x2_full = pickle.load(f)
    
    # Filter by mechanism
    x1_filtered, y_filtered = filter_by_mechanism(x1_full, y_full, mechanism_ids)
    
    # Handle x2 filtering based on data type
    if isinstance(x2_full, dict):
        # Nature test data with nested dictionary
        x2_filtered = filter_test_data(x2_full, y_full, mechanism_ids)
    else:
        # Standard format
        x2_filtered, _ = filter_by_mechanism(x2_full, y_full, mechanism_ids)
    
    print(f"  - Loaded: x1={x1_filtered.shape}, x2={x2_filtered.shape if not isinstance(x2_filtered, dict) else 'nested_dict'}, y={y_filtered.shape}")
    
    # Clear full data from memory
    del x1_full, x2_full, y_full
    gc.collect()
    
    # Create dataset
    dataset = NatureSetDataset(
        x1=x1_filtered,
        x2=x2_filtered,
        y=y_filtered,
        n_min=n_min,
        n_max=n_max,
        samples_per_epoch=samples_per_epoch,
        is_nature_test=is_nature_test,
        nature_test_timepoints=nature_test_timepoints,
        nature_test_error=nature_test_error,
        noise_enabled=noise_enabled,
        noise_std_dev=noise_std_dev,
        feature_config=feature_config
    )
    
    # Create DataLoader
    # Note: pin_memory=False to reduce memory pressure with large datasets
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=scatter_set_collate_fn,
        num_workers=num_workers,
        pin_memory=False,  # Disabled to reduce memory usage
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive
    )
    
    print(f"  Created {split} dataloader with {len(loader)} batches\n")
    
    # Return metadata (label mapping, etc.)
    metadata = {
        'label_mapping': dataset.inverse_label_map,
        'n_samples': dataset.n_samples,
        'n_profiles': dataset.n_profiles,
        'n_timepoints': dataset.n_timepoints,
        'feature_dim': dataset.feature_dim
    }
    
    return loader, metadata


def test_dataloader():
    """Test the dataloader functionality with the simplified API."""
    print("Testing NatureSetDataset and DataLoader...")
    print()
    
    # Load configuration
    config = load_config('config/train_set_transformer.yaml')
    data_dir = config.get('data_dir', 'data/nature_data')
    mechanisms = config['mechanisms']
    is_nature_test = config.get('is_nature_test', True)
    nature_test_timepoints = config.get('nature_test_timepoints', 20)
    nature_test_error = config.get('nature_test_error', 1)
    n_min = config['set_generation']['n_min']
    n_max = config['set_generation']['n_max']
    batch_size = min(8, int(config['training']['batch_size']))
    num_workers = 0
    samples_per_epoch = 32
    
    # Get noise parameters
    noise_config = config['set_generation'].get('noise', {})
    noise_enabled = noise_config.get('enabled', False)
    noise_std_dev = noise_config.get('std_dev', 0.01)
    
    # Get feature configuration
    feature_config = config['set_generation'].get('features', None)
    
    print("="*80)
    print("TESTING DATALOADERS")
    print("="*80)
    print()
    
    # Create train dataloader
    train_loader, train_metadata = create_dataloader(
        data_dir=data_dir,
        mechanisms=mechanisms,
        split='train',
        n_min=n_min,
        n_max=n_max,
        batch_size=batch_size,
        num_workers=num_workers,
        samples_per_epoch=samples_per_epoch,
        noise_enabled=noise_enabled,
        noise_std_dev=noise_std_dev,
        feature_config=feature_config
    )
    
    # Create val dataloader
    val_loader, val_metadata = create_dataloader(
        data_dir=data_dir,
        mechanisms=mechanisms,
        split='val',
        n_min=n_min,
        n_max=n_max,
        batch_size=batch_size,
        num_workers=num_workers,
        noise_enabled=noise_enabled,
        noise_std_dev=noise_std_dev,
        feature_config=feature_config
    )
    
    # Create test dataloader
    test_loader, test_metadata = create_dataloader(
        data_dir=data_dir,
        mechanisms=mechanisms,
        split='test',
        n_min=n_min,
        n_max=n_max,
        batch_size=batch_size,
        num_workers=num_workers,
        is_nature_test=is_nature_test,
        nature_test_timepoints=nature_test_timepoints,
        nature_test_error=nature_test_error,
        noise_enabled=False,  # No noise for test data
        feature_config=feature_config
    )
    
    # Test a batch from training loader
    print("Testing training loader:")
    for features, mask, labels in train_loader:
        print(f"  - Features shape: {features.shape}")
        print(f"  - Mask shape: {mask.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Feature range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"  - Unique labels: {torch.unique(labels).tolist()}")
        print(f"  - Number of points per sample: {mask.sum(dim=1).tolist()[:5]}...")
        break
    
    print()
    print("Testing validation loader:")
    for features, mask, labels in val_loader:
        print(f"  - Features shape: {features.shape}")
        print(f"  - Mask shape: {mask.shape}")
        print(f"  - Labels shape: {labels.shape}")
        break
    
    print()
    print("Testing test loader:")
    for features, mask, labels in test_loader:
        print(f"  - Features shape: {features.shape}")
        print(f"  - Mask shape: {mask.shape}")
        print(f"  - Labels shape: {labels.shape}")
        break
    
    print()
    print(f"Label mapping: {train_metadata['label_mapping']}")
    print()
    print("All tests passed!")


if __name__ == "__main__":
    test_dataloader()