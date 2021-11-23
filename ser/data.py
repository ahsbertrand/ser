
from torch.utils.data import DataLoader 
from torchvision import datasets   

def load_data(directory, download_bool, train_bool, ts, batch_size, shuffle_bool, num_workers=1):
    
    dataloader = DataLoader(
        datasets.MNIST(root=directory, download=download_bool, train=train_bool, transform=ts),
    batch_size=batch_size,
    shuffle=shuffle_bool,
    num_workers=num_workers,
)

    return dataloader