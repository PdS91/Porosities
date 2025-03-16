import torch
from torch.utils.data import Dataset, random_split
import numpy as np
from Lib.Data import PorosityDistribution, extract_microstructures, extract_porosities_points
from Lib.Tools import conditioned_random_sampling

class MicrostructureDataset(Dataset):
    def __init__(self, sample_path, train=True, val=False, test=False,keep_doubles=True, split_ratio=[0.8, 0.1, 0.1], device = 'cpu',seed=42):
        self.extracted_data = extract_microstructures(sample_path,keep_density_doubles=keep_doubles)
        self.samples = list(self.extracted_data.keys())
        self.device = device

        # Set random seed for reproducibility
        np.random.seed(seed)
        np.random.shuffle(self.samples)

        # Split into train, val, test
        total_samples = len(self.samples)
        train_size = int(split_ratio[0] * total_samples)
        val_size = int(split_ratio[1] * total_samples)
        test_size = total_samples - train_size - val_size

        if train:
            self.samples = self.samples[:train_size]
        elif val:
            self.samples = self.samples[train_size:train_size + val_size]
        elif test:
            self.samples = self.samples[train_size + val_size:]
        else:
            raise ValueError("Invalid split type. Use train=True, val=True, or test=True.")

    def __len__(self):
        return len(self.samples)  # Number of samples

    def __getitem__(self, idx):
        sample_number = self.samples[idx]  # Get sample number
        porosity_distribution = self.extracted_data[sample_number]  # Get PorosityDistribution object
        data = porosity_distribution.as_3Darray()  # Get data as 3D array

        # Convert to PyTorch tensor
        data = torch.from_numpy(data).float()
        data = data.permute(3, 0, 1, 2)  # Permute dimensions to (4, 30, 30, 30)

        # Get target (density)
        target = porosity_distribution.density
        target = torch.tensor(target).float()


        return data.to(self.device), target.to(self.device)
    
class PorosityDataset(Dataset):
    def __init__(self, sample_path, samples_per_density = 1000, train=True, val=False, test=False,keep_doubles=True, split_ratio=[0.8, 0.1, 0.1], device = 'cpu',seed=42):
        filt_extracted_porosities, density_set = extract_porosities_points(sample_path,keep_density_doubles=False)
        balanced_porosities = conditioned_random_sampling(filt_extracted_porosities,n_samples=samples_per_density)
        self.extracted_data = balanced_porosities
        
        self.samples = list(self.extracted_data.index)
        self.device = device

        # Set random seed for reproducibility
        np.random.seed(seed)
        np.random.shuffle(self.samples)

        # Split into train, val, test
        total_samples = len(self.samples)
        train_size = int(split_ratio[0] * total_samples)
        val_size = int(split_ratio[1] * total_samples)
        test_size = total_samples - train_size - val_size

        if train:
            self.samples = self.samples[:train_size]
        elif val:
            self.samples = self.samples[train_size:train_size + val_size]
        elif test:
            self.samples = self.samples[train_size + val_size:]
        else:
            raise ValueError("Invalid split type. Use train=True, val=True, or test=True.")

    def __len__(self):
        return len(self.samples)  # Number of samples

    def __getitem__(self, idx):
        sample_number = self.samples[idx]  # Get sample number
        sample = self.extracted_data.loc[sample_number]  # Get PorosityDistribution object
        sample = sample.values[:-1]

        # Convert to PyTorch tensor
        data = torch.from_numpy(sample[:-2]).float()

        # Get target (density)
        target = torch.tensor(sample[-1]).float()


        return data.to(self.device), target.to(self.device)


class PorosityDataset2(Dataset):
    def __init__(self, sample_path, samples_per_density = 1000, train=True, val=False, test=False,keep_doubles=True, split_ratio=[0.8, 0.1, 0.1], device = 'cpu',seed=42):
        filt_extracted_porosities, density_set = extract_porosities_points(sample_path,keep_density_doubles=False)
        balanced_porosities = conditioned_random_sampling(filt_extracted_porosities,n_samples=samples_per_density,random_pointer=3)
        self.extracted_data = balanced_porosities
        
        self.samples = list(self.extracted_data.index)
        self.device = device

        # Set random seed for reproducibility
        np.random.seed(seed)
        np.random.shuffle(self.samples)

        # Split into train, val, test
        total_samples = len(self.samples)
        train_size = int(split_ratio[0] * total_samples)
        val_size = int(split_ratio[1] * total_samples)
        test_size = total_samples - train_size - val_size

        if train:
            self.samples = self.samples[:train_size]
        elif val:
            self.samples = self.samples[train_size:train_size + val_size]
        elif test:
            self.samples = self.samples[train_size + val_size:]
        else:
            raise ValueError("Invalid split type. Use train=True, val=True, or test=True.")

    def __len__(self):
        return len(self.samples)  # Number of samples

    def __getitem__(self, idx):
        sample_number = self.samples[idx]  # Get sample number
        sample = self.extracted_data.loc[sample_number]  # Get PorosityDistribution object
        sample = sample.values

        # Convert to PyTorch tensor
        data = torch.from_numpy(sample[-3:]).float()

        # Get target (density)
        target = torch.from_numpy(sample[:3]).float()
        
        condition = torch.tensor(sample[4]).float()


        return data.to(self.device), target.to(self.device), condition.to(sample)