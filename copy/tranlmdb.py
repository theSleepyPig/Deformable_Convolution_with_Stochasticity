import os
import lmdb
import pickle
from tqdm import tqdm
from torchvision import datasets, transforms

def dataset_to_lmdb(dataset, lmdb_path):
    """Converts a PyTorch dataset to LMDB format."""
    # Initialize LMDB
    map_size = 1 << 40  # 1TB, adjust as needed
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    with env.begin(write=True) as txn:
        # Store dataset length
        txn.put(b'__len__', pickle.dumps(len(dataset)))
        
        for idx, (img, label) in enumerate(tqdm(dataset)):
            # Serialize sample
            sample = (img.numpy(), label)
            txn.put(str(idx).encode('ascii'), pickle.dumps(sample))
    
    print(f"Dataset saved to {lmdb_path}")

def main():
    # Define your dataset
    data_dir = "~/datasets/CIFAR/"
    
    # Define transformations (modify if needed)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Load CIFAR-10 or CIFAR-100 dataset
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # Define LMDB output paths
    train_lmdb_path = os.path.join(data_dir, "train.lmdb")
    test_lmdb_path = os.path.join(data_dir, "test.lmdb")

    # Convert datasets
    print("Converting training dataset...")
    dataset_to_lmdb(train_dataset, train_lmdb_path)

    print("Converting testing dataset...")
    dataset_to_lmdb(test_dataset, test_lmdb_path)

if __name__ == "__main__":
    main()