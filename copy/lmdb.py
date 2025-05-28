import os
import lmdb
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from torchvision import datasets, transforms
from tqdm import tqdm

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(str(idx).encode('ascii'))
            img, label = pickle.loads(byteflow)
        return torch.tensor(img), label

def dataset_to_lmdb(dataset, lmdb_path):
    map_size = 1 << 40  # 1TB, adjust as needed
    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(len(dataset)))

        for idx, (img, label) in enumerate(tqdm(dataset)):
            sample = (img.numpy(), label)
            txn.put(str(idx).encode('ascii'), pickle.dumps(sample))

def get_lmdb_loaders(data_dir, batch_size, worker):
    train_lmdb_path = f"{data_dir}/train.lmdb"
    test_lmdb_path = f"{data_dir}/test.lmdb"

    train_dataset = LMDBDataset(train_lmdb_path)
    test_dataset = LMDBDataset(test_lmdb_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=worker)

    return train_loader, test_loader

def convert_cifar10_to_lmdb(data_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_lmdb_path = os.path.join(data_dir, "train.lmdb")
    test_lmdb_path = os.path.join(data_dir, "test.lmdb")

    print("Converting CIFAR-10 training set to LMDB...")
    dataset_to_lmdb(train_dataset, train_lmdb_path)

    print("Converting CIFAR-10 testing set to LMDB...")
    dataset_to_lmdb(test_dataset, test_lmdb_path)

# Replace the existing data loader with the LMDB loader
train_loader, test_loader = get_lmdb_loaders(args.data_dir, args.batch_size, args.worker)

# Call the conversion function if needed
convert_cifar10_to_lmdb(args.data_dir)

# Proceed with the rest of the training loop as before