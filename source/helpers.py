import numpy as np
import torch
import os
import glob
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2

from sklearn.model_selection import train_test_split 

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils

# helper functions
def process_dicom_image(path: str, resize=True) -> np.ndarray:
    """ Given a path to a DICOM image, process and return the image. 
        Reduces the size in memory.
    """
    # TODO: add functionality to read from s3 NOT local
#     s3_client = boto3.client('s3')
#     obj = s3_client.get_object(Bucket=sagemaker_bucket, Key=s3_image_path)
#     s3_test_dicom = pydicom.dcmread(io.BytesIO(obj['Body'].read()))

    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array
    image = image - np.min(image)
    
    if np.max(image) != 0:
        image = image / np.max(image)
    
    image = (image * 255).astype(np.uint8)
    # resize the image 256px
    if resize:
        image = cv2.resize(image, (256,256))
#     print(image.shape)
#     print(type(image))
    return image

def get_sequence_images(path: str) -> list():
    """ Returns a sorted list of images from a MRI sequence subfolder. 
        Excludes images that have no image. i.e. - only black.
    
        Arguments:
            path (str): path to a MRI sequence folder. ex. ./train/00000/FLAIR
        Returns:
            images (list): List of np.ndarray images 
    """
    images = []
    image_path_list = glob.glob(path + '/*') # at the MRI sequence level
    # sort the path list in place by image number 
    image_path_list.sort(key=lambda x: int(x.split('/')[-1].split('-')[-1].split('.')[0]))
    
    for p in image_path_list:
        img = process_dicom_image(p)
        # only add if there is an visual image. i.e. - if it is not black
        if np.max(img) == 0:
            continue
        images.append(img)
        
    return images

def get_middle_image(path: str) -> np.ndarray:
    """ Returns the middle image in a sequence of MRI images. Removes
        images that are only black.
        
        Arguments:
            path (str): path to a MRI sequence folder. ex. ./train/00000/FLAIR
        Returns:
            images (np.ndarray)
    """
    image_path_list = glob.glob(path + '/*') # at the MRI sequence level
    image_path_list.sort(key=lambda x: int(x.split('/')[-1].split('-')[-1].split('.')[0]))
    idx = len(image_path_list)//2
    image = process_dicom_image(image_path_list[idx])
    
    return image

def train_val_split(path: str, val_ratio=0.10):
    """ Splits the train_labels.csv file into training and validation
        dataframes to be used in the DataSet Class.
        
        Arguments:
            path (str): path to the data folder
            val_ratio (float): ratio to split for validation
        
        Returns:
            train_df, val_df (pd.DataFrame): labels for training and validation
    """
    # read in train_labels.csv file
    train_labels_df = pd.read_csv(os.path.join(path, 'train_labels.csv'))

    # drop [00109, 00123, 00709]
    train_labels_df.drop(train_labels_df.loc[train_labels_df['BraTS21ID']==109].index, inplace=True)
    train_labels_df.drop(train_labels_df.loc[train_labels_df['BraTS21ID']==123].index, inplace=True)
    train_labels_df.drop(train_labels_df.loc[train_labels_df['BraTS21ID']==709].index, inplace=True)
    
    # separate into two dataframes. 
    mask = train_labels_df['MGMT_value'] == 1
    df_pos = train_labels_df[mask] # MGMT_value == 1
    df_neg = train_labels_df[~mask] # MGMT_value == 0

    # use scikit-learn to split each pos/neg DataFrame into train/val
    train_pos, val_pos = train_test_split(df_pos, test_size=val_ratio)
    train_neg, val_neg = train_test_split(df_neg, test_size=val_ratio)

    # concatenate the pos with the negative
    train_df = pd.DataFrame(pd.concat([train_pos, train_neg])).sort_values(by='BraTS21ID')
    train_df['Split'] = 'train'
#     print('Train labels dataframe length: ', len(train_df))
    val_df = pd.DataFrame(pd.concat([val_pos, val_neg])).sort_values(by='BraTS21ID')
    val_df['Split'] = 'valid'
#     print('Validation labels dataframe length: ', len(val_df))

    assert(len(train_df) + len(val_df) == len(train_labels_df))
    
    return train_df, val_df

# helper function to process DICOM images. 
def get_patient_images(path):
    """ Process all images in a patient subfolders FLAIR, T1w, T1wCE, T2w. Sorts images and returns
        an array of images of size (W,H,N,C) where:
            W=image width reduced from 512 to 256 px
            H=image height reduced from 512 to 256 px
            N=number if images in each MRI sequence. 
            C=number if MRI seqence types
        If the number of images in each subset is less than 128 the remaining images are created 
        as np.array with pixel values == 0.
        
        Parameters:
            path (str): patient id path ./train/00000
        Returns:
            np.array of size (256, 256, 128, 4)
    """
    seq_len = 64

    # get the path of MRI seq subfolder
    flair_path = os.path.join(path, 'FLAIR')
    t1w_path = os.path.join(path, 'T1w')
    t1wce_path = os.path.join(path, 'T1wCE')
    t2w_path = os.path.join(path, 'T2w')
    
    # get the images in each sequence
    # FLAIR
    flair_imgs = get_sequence_images(flair_path)
    if len(flair_imgs) >= seq_len:
        start = (len(flair_imgs)//2)-int(seq_len/2)
        end = (len(flair_imgs)//2)+int(seq_len/2)
        flair_imgs = np.array(flair_imgs[start:end]).T
    else:
        diff = seq_len - len(flair_imgs)
        flair_imgs = np.concatenate((np.array(flair_imgs).T, np.zeros((256,256,diff))),axis=-1)
    
    # T1w
    t1w_imgs = get_sequence_images(t1w_path)
    if len(t1w_imgs) >= seq_len:
        start = (len(t1w_imgs)//2)-int(seq_len/2)
        end = (len(t1w_imgs)//2)+int(seq_len/2)
        t1w_imgs = np.array(t1w_imgs[start:end]).T
    else:
        diff = seq_len - len(t1w_imgs)
        t1w_imgs = np.concatenate((np.array(t1w_imgs).T, np.zeros((256,256,diff))),axis=-1)
        
    # T1wCE
    t1wce_imgs = get_sequence_images(t1wce_path)
    if len(t1wce_imgs) >= seq_len:
        start = (len(t1wce_imgs)//2)-int(seq_len/2)
        end = (len(t1wce_imgs)//2)+int(seq_len/2)
        t1wce_imgs = np.array(t1wce_imgs[start:end]).T
    else:
        diff = seq_len - len(t1wce_imgs)
        t1wce_imgs = np.concatenate((np.array(t1wce_imgs).T, np.zeros((256,256,diff))),axis=-1)
        
    # T2w
    t2w_imgs = get_sequence_images(t2w_path)
    if len(t2w_imgs) >= seq_len:
        start = (len(t2w_imgs)//2)-int(seq_len/2)
        end = (len(t2w_imgs)//2)+int(seq_len/2)
        t2w_imgs = np.array(t2w_imgs[start:end]).T
    else:
        diff = seq_len - len(t2w_imgs)
        t2w_imgs = np.concatenate((np.array(t2w_imgs).T, np.zeros((256,256,diff))),axis=-1)
    
    return np.moveaxis(np.array((flair_imgs, t1w_imgs, t1wce_imgs, t2w_imgs)), 0, -1)

# PyTorch Custom Dataset to be used in DataLoader
class BrainScanDataset(Dataset):
    """ MRI brain scan dataset. """
    def __init__(self, data_dir, split='train', val_ratio=0.10, transform=None):
        """
            Args:
             data_dir (str): Path to the data folder
             split (str): 'train' or 'valid' 
             transform (bool): Apply transforms
        """
        self.data_dir = data_dir
        
        # get training labels
        t_df, v_df = train_val_split(data_dir, val_ratio=val_ratio)
        
        if split == 'train':
            labels_df = t_df
        elif split == 'valid':
            labels_df = v_df
            
        label_id = labels_df[labels_df.columns[0]] # BraTS21ID
        label_y = labels_df[labels_df.columns[1]] # MGMT_value
        self.labels_dict = {str(l_id).zfill(5): y for l_id, y in zip(label_id, label_y)}
        
        # TODO: Correct for Testing and Training
        self.data_path = os.path.join(data_dir, 'train')
        
        # get patient ids
        self.id_path_list = [path for path in sorted(glob.glob(self.data_path + '/*')) 
                             if path.split('/')[-1] in self.labels_dict]
        self.id_list = [path.split('/')[-1] for path in sorted(glob.glob(self.data_path + '/*'))
                        if path.split('/')[-1] in self.labels_dict]
        
        # TODO: image transforms
        self.transform = transform
        
    def __len__(self):
        return len(self.id_path_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        images = get_patient_images(self.id_path_list[idx])
        labels = self.labels_dict[self.id_list[idx]]
        
        imgs_tensor = torch.tensor(images, dtype=torch.float32).permute(-1, 0, 1, 2) # need to reshape
#         print(imgs_tensor.shape)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return imgs_tensor, labels_tensor