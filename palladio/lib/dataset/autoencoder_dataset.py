import os
from PIL import Image

import torch

import pandas as pd


# 🖼️🔄🖼️
class AutoencoderDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path, images_folder, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df["category_enc"] = self.df["category"].astype('category').cat.codes
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        filename = self.df.iloc[index]["image_name"]
        # label = torch.tensor(self.df.iloc[index]["category_enc"], dtype=torch.long)

        if not os.path.isfile(filename):
            image = Image.open(
                os.path.join(self.images_folder, filename)).convert('RGB')
        else:
            pass

        if self.transform is not None:
            image = self.transform(image)

        # return image, label
        return image, image
