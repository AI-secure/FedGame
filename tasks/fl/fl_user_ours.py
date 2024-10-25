from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader



@dataclass
class FLUserOurs:
    user_id: int = 0
    compromised: bool = False
    train_loader: DataLoader = None
    test_loader: DataLoader = None
