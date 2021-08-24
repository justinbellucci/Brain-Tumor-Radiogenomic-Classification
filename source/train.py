import argparse
import json
import os
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# import EfficientNet-3D model from https://github.com/shijianjian/EfficientNet-PyTorch-3D
from efficientnet_pytorch_3d import EfficientNet3D

from helpers import BrainScanDataset

# sagemaker specific
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EfficientNet3D.from_name(model_name = model_info['model_name'], 
                                     override_params = model_info['output_dim'], 
                                     in_channels = model_info['input_dim']) 

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, data_dir, val_ratio):
    print("Getting training and validation data loaders...")

    # create training and validation datasets
    train_dataset = BrainScanDataset(data_dir=data_dir, split='train', val_ratio=val_ratio)
    valid_dataset = BrainScanDataset(data_dir=data_dir, split='valid', val_ratio=val_ratio)
    
    # create training and validation DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    print("Done getting dataloaders!")
    return train_loader, valid_loader

# training function
def train(model, train_loader, valid_loader, epochs, criterion, optimizer, device, writer):
    """ This is the training method that is called by the PyTorch training script. 
    
        Parameters:
            model        - The PyTorch model that we wish to train.
            train_loader - The PyTorch DataLoader that should be used during training.
            valid_loader - The PyTorch DataLoader that should be used during validation.
            epochs       - The total number of epochs to train for.
            criterion    - The loss function used for training. 
            optimizer    - The optimizer to use during training.
            device       - Where the model and data should be loaded (gpu or cpu).
            writer       - SummaryWriter instance for logging model data to Tensorboard
    """
    
    for epoch in range(1, epochs + 1):
        
        # ----- Training pass -----
        model.train() # Make sure that the model is in training mode.

        train_loss = 0.0
        valid_loss = 0.0
        correct = 0.0
        total = 0.0
        
        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # get predictions from model
            outputs = model(batch_x)
            
            # perform backprop
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # record training metrics
            train_loss += loss.item()
            writer.add_scalar("Running Loss - Train", train_loss, epoch)
            
        # ----- Validation pass -----
        model.eval()
        
        for batch_x, batch_y in valid_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
        
            # record validation metrics
            _, preds = torch.max(outputs, 1) # may need outputs.data
#             total += batch_y.size(0)
#             correct += (preds == batch_y).sum().item()
            
            valid_loss += loss.item()
            writer.add_scalar("Running Loss - Valid", valid_loss, epoch)
#             writer.add_pr_curve('PR_curve', batch_y.item(), preds.item())
         
        print('\nOutputs shape: ', outputs.shape)
        print('Preds shape:', preds.shape)
        print('Batch_y size:', batch_y.size(0))
        print('Batch_y shape:', batch_y.shape)
        
        train_loss = train_loss / len(train_loader)
        valid_loss = train_loss / len(valid_loader)
#         accuracy = 100 * correct / total
        
        # record training/validation metrics for each epoch
        print("Epoch: {} -- Training Loss: {:.5f} -- Validation Loss: {:.5f}".format(
            epoch, train_loss, valid_loss))
#         print("Epoch: {} -- Training Loss: {:.5f} -- Validation Loss: {:.5f} -- Accuracy: {}".format(
#             epoch, train_loss, valid_loss, accuracy))
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Validation Loss', valid_loss, epoch)
#         writer.add_scalar('Accuracy', accuracy, epoch)
        

if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters
    # directories for training data and saving models; 
    # set automatically - Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING']) 
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=3, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--input_dim', type=int, default=4, metavar='IN',
                        help='number of input dimensions to model (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--val_ratio', type=float, default=0.15, metavar='VR',
                        help='validation dataset ratio (default: 0.15)')
    
    # args holds all passed-in arguments
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)
    
    writer = SummaryWriter() # for recording model data to Tensorboard 
    
    # Load the training data.
    train_loader, valid_loader = _get_train_data_loader(args.batch_size, args.data_dir, args.val_ratio)

    # instantiate model with input arguments
    model = EfficientNet3D.from_name(model_name = 'efficientnet-b5', 
                                     override_params = {'num_classes': 2}, 
                                     in_channels = args.input_dim)
    
    model.to(device) # move model to GPU if available, else CPU

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    criterion = nn.CrossEntropyLoss() 
#     criterion = nn.BCELoss()
    
    # Trains the model (given line of code, which calls the above training function)
    train(model, train_loader, valid_loader, args.epochs, criterion, optimizer, device, writer)
    
    writer.flush() # make sure that all pending events have been written to disk
    
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')

    with open(model_info_path, 'wb') as f:
        model_info = {
            'model_name': 'efficientnet-b5',
            'input_dim': args.input_dim,
            'output_dim': {'num_classes': 2},
            'epochs': args.epochs,
            'val_ratio': args.val_ratio,
            'lr': args.lr
        }
        torch.save(model_info, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
    
    writer.close() # close the writer since we are done using it