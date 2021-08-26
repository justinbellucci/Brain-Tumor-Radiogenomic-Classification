import argparse
import json
import os
import pandas as pd
import io

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# import EfficientNet-3D model from https://github.com/shijianjian/EfficientNet-PyTorch-3D
from efficientnet_pytorch_3d import EfficientNet3D

# from helpers import BrainScanTestDataset

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

################ TODO ###############

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

################ TODO ###############

def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    