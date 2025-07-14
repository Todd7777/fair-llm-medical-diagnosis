import fundus_image_toolbox as fit
import numpy as np
from PIL import Image


# all for transform parameter of PyTorch dataloader
class chest_xray_img_processor:
    # process img similar to below
    pass


class pathology_img_processor:
    # process img similar to below
    pass


class fundus_img_processor:
    # changes to the below code depend on what format is inputted into VLM, PIL image, np array, etc
    # currently returns np array

    def __init__(self, target_size=(224, 224), normalize=True, augment=False):
        self.target_size = target_size
        self.normalize = normalize
        self.augment = False

    def crop_img(self, img_path):
        return fit.crop(img_path, self.target_size)

    def resize(self, img):
        # img is expected to be a PIL Image object
        resized_img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        return resized_img

    def normalize_img(self, img):
        return np.array(img).astype(np.float32) / 255.0

    def augment_img(self, img):
        # flip, rotate, etc.)
        pass

    def img_process(self, img_path):
        img = self.crop_img(img_path)
        img = self.resize(img)
        if self.normalize:
            img = self.normalize_img(img)
        if self.augment:
            img = self.augment_img(img)
        return img
