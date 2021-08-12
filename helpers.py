import numpy as np


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