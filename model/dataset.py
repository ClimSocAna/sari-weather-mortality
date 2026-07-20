import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """
    Step-count dataset for Lightning training loops.

    The model loads and owns the full training tensors itself; this dataset only
    controls how many optimizer/validation steps Lightning runs per epoch.
    """

    def __init__(self, num_steps: int = 100) -> None:
        """
        Args:
            num_steps: Number of Lightning steps in this dataset
        """
        self.num_steps = num_steps

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(0)
