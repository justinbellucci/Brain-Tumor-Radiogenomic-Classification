import numpy as np
import torch

# helper functions
def process_dicom_image(path: str) -> np.ndarray:
    """ Given a path to a DICOM image, process and return the image. 
        Reduces the size in memory.
    """
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array
    image = image - np.min(image)
    
    if np.max(image) != 0:
        image = image / np.max(image)
    
    image = (image * 255).astype(np.uint8)
        
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

################ UPDATE ################
# create a PyTorch Custom Dataset to be used in DataLoader
class BrainScanDataset(Dataset):
    """ MRI brain scan dataset. """
    def __init__(self, data_dir, transform=None):
        """
            Args:
             data_dir (str): Path to the data folder
             train (str): 'train' or 'test' depending on what you want
             transform (bool): Apply transforms
        """
        self.data_dir = data_dir
        
        # get training labels
        train_labels_df = pd.read_csv(os.path.join(self.data_dir, 'train_labels.csv'))
        label_id = train_labels_df[train_labels_df.columns[0]] # BraTS21ID
        label_y = train_labels_df[train_labels_df.columns[1]] # MGMT_value
        self.train_labels = {str(l_id).zfill(5): y for l_id, y in zip(label_id, label_y)}
        
        # TODO: Correct for Testing and Training
        self.train_path = os.path.join(data_dir, 'train')
        
        # get patient ids
        self.id_path_list = [path for path in sorted(glob.glob(self.train_path + '/*'))]
        self.id_list = [path.split('/')[-1] for path in sorted(glob.glob(self.train_path + '/*'))]
        
        # TODO: Remove [00109, 00123, 00709] if they exist
        
        # TODO: image transforms
        self.transform = transform
        
    def __len__(self):
        return len(self.id_path_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        images = get_patient_images(self.id_path_list[idx])
        labels = self.train_labels[self.id_list[idx]]
        
        imgs_tensor = torch.tensor(images, dtype=torch.float32).permute(-1, 0, 1, 2) # need to reshape
        print(imgs_tensor.shape)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return imgs_tensor, labels_tensor