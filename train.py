#!usr/bin/bash python

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.FullyConnected import FullyConnected
from data_loading.load_dataset import get_dataloaders

def get_accuracy(model, data_loader, device):
  model.eval()
  correct, total = 0, 0
  for data, labels in data_loader:
    data = data.to(device)
    labels = labels.to(device)
    total += labels.shape[0]
    with torch.no_grad():
      preds = model(data).argmax(dim=-1)
      correct += (preds == labels).sum().item()
  return correct/total


def train_model_with_logger(model, optimizer, train_loader, val_loader, loss_fn, device, num_epochs=40, logging_dir='runs/act_fn_experiment'):
    # Create TensorBoard logger
    writer = SummaryWriter(logging_dir)

    # Set model to train mode
    model.train()

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for x, y in train_loader:

            ## Step 1: Move input data to device
            x = x.to(device)
            y = y.to(device)

            ## Step 2: Run the model on the input data
            preds = model(x)

            ## Step 3: Calculate the loss
            loss = loss_fn(preds, y)

            ## Step 4: Perform backpropagation
            optimizer.zero_grad()
            loss.backward()

            ## Step 5: Update the parameters
            optimizer.step()

            ## Step 6: Take the running average of the loss
            epoch_loss += loss.item()

        # Add average loss to TensorBoard
        epoch_loss /= len(train_loader)
        writer.add_scalar('training_loss',
                          epoch_loss,
                          global_step = epoch + 1)
        val_acc = get_accuracy(model, val_loader, device)
        writer.add_scalar('validation_acc',
                          val_acc,
                          global_step = epoch + 1)

    writer.close()

dataset_options = ['mnist']
model_options = ["fc"]
act_fn_options = ["sigmoid", "tanh", "relu", "leakyrelu", "elu", "swish", "all"]

model_table = {"fc": FullyConnected}

def main():
  # Define the ArgumentParser
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='mnist', choices=dataset_options)
  parser.add_argument('--model', default='fc', choices=model_options)
  parser.add_argument('--act_fn', type=str, required=True, choices=act_fn_options)
  parser.add_argument('--epochs', type=int, default=40)
  parser.add_argument('--lr', type=float, default=0.1)
  parser.add_argument('--batch_size', type=int, default=64)
  args = parser.parse_args()

  # Dictionary of useable activation functions, feel free to add more
  # Activations found here: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
  # Consider also trying functional: https://pytorch.org/docs/stable/nn.functional.html
  act_fn_by_name = {
      "sigmoid": [nn.Sigmoid],
      "tanh": [nn.Tanh],
      "relu": [nn.ReLU],
      "leakyrelu": [nn.LeakyReLU],
      "elu": [nn.ELU],
      "swish": [nn.SiLU]
  }
  act_fn_to_name = {value[0]: key for key,value in act_fn_by_name.items()}
  act_fn_by_name["all"] = [act_fn[0] for act_fn in act_fn_by_name.values()]
  if args.act_fn not in act_fn_by_name:
    raise ValueError(f"Invalid activation {args.act_fn}\n \
                      Valid activations: {' '.join([key for key in act_fn_by_name.keys()])}")

  # Select Device
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  # Initialize data loaders & get # of classes
  num_classes, train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, device, path="../data", flatten=True)

  # Select model class
  model_type = model_table[args.model]

  loss_fn = F.cross_entropy

  # Train a model for each activation function & record loss
  for act_fn in act_fn_by_name[args.act_fn]:
    print(act_fn)
    model = model_type(10, 784, act_fn)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    train_model_with_logger(model, optimizer, train_loader, test_loader, loss_fn, device, num_epochs=args.epochs, logging_dir=f'runs/act_fn_experiment/{act_fn_to_name[act_fn]}')


if __name__ == '__main__':
    main()
