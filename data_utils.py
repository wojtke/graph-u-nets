import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.loader import DataLoader


def get_dataset_cv_incicies(dataset, k_folds=10):
    """Returns a list of train and test indicies for each fold.

    Args:
        dataset (torch_geometric.data.Dataset): Dataset to split into folds.
        k_folds (int): Number of folds to split the dataset into (default: 10).
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    folds = skf.split(np.arange(dataset.len()), [data.y.item() for data in dataset])

    return folds


def get_dataloaders(dataset, batch_size=32, val_size=0.1):
    """Returns train and validation dataloaders for a dataset.

    Args:
        dataset (torch_geometric.data.Dataset): Dataset to split into train and validation.
        batch_size (int): Batch size for the dataloaders (default: 32).
        val_size (float): Proportion of the dataset to use for validation (default: 0.1).
    """
    train_idx, val_idx = train_test_split(
        np.arange(dataset.len()),
        test_size=val_size,
        stratify=[data.y.item() for data in dataset],
    )
    train_loader = DataLoader(
        dataset[list(train_idx)], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        dataset[list(val_idx)], batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader
