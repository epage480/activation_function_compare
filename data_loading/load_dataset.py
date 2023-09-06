#!usr/bin/bash python
import torch
from torchvision import datasets, transforms


def get_dataloaders(dataset_name, batch_size, device, path="../data", flatten=False):
  # Table providing normalization parameters for each dataset
  normalize_table = {"mnist": transforms.Normalize((0.1307,), (0.3081,)),
                   "cifar10": transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))}

  # Define basic train/test arguments
  train_kwargs = {"batch_size": batch_size, "shuffle": True}
  test_kwargs = {"batch_size": batch_size, "shuffle": False}
  if device == "cuda":
      cuda_kwargs = {'num_workers': 4,
                      'pin_memory': True}
      train_kwargs.update(cuda_kwargs)
      test_kwargs.update(cuda_kwargs)

  # Define the transformations to apply to dataset in dataloader
  transformList = [transforms.ToTensor()]
  if dataset_name in normalize_table: transformList.append(normalize_table[dataset_name])
  else: print(f"WARNING: no normalization parameters found for {dataset_name}")
  if flatten: transformList.append(transforms.Lambda(lambda x: torch.flatten(x)))
  transform = transforms.Compose(transformList)

  # Load Datasets
  if dataset_name == "mnist":
    num_classes = 10
    train_dataset = datasets.MNIST(path, train=True, download=True,
                          transform=transform)
    test_dataset = datasets.MNIST(path, train=False, download=True,
                        transform=transform)
  else:
    print("Dataset not defined")

  # Initialize data loaders
  train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
  test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

  return num_classes, train_loader, test_loader
