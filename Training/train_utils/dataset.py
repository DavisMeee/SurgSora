import os, io, csv, math, random
import numpy as np
from einops import rearrange
import pandas as pd

import torch
from decord import VideoReader, cpu
import torch.distributed as dist

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import ast

def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)



class ESDDataset(Dataset):
    def __init__(
            self,
            meta_path='./dataset/condition_train_dual.csv',
            data_dir='./dataset/result_frame',
            sample_size=[384,384], 
            sample_stride=1, 
            sample_n_frames=14,
    ):
        zero_rank_print(f"loading annotations from {meta_path} ...")

        metadata = pd.read_csv(meta_path)
        metadata['caption'] = metadata['name']
        del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)
        self.data_dir = data_dir

        self.length = len(self.metadata)
        # print(f"number of data: {self.length}")

        self.sample_stride = sample_stride
        # print(f"sample stride: {self.sample_stride}")
        self.sample_n_frames = sample_n_frames


        self.sample_size = sample_size

        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    """relative video path and full video path. Modified accordingly"""
    """Not used for our dataset"""

    def _get_video_path(self, sample):
        rel_video_fp = str(sample["videoid"]).rjust(6, "0") + "_" + str(sample["clipid"]).rjust(4, "0")  # not important
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        # print(f"full video path: {full_video_fp}")
        return full_video_fp, rel_video_fp

    def get_batch(self, index):

        # No need of while loop
        index = index % len(self.metadata)
        sample = self.metadata.iloc[index]
        video_path, rel_path = self._get_video_path(sample)

        import cv2
        def read_and_stack_images(ful_path):
            """Reads images from a folder and stacks them into a numpy array."""
            image_list = []
            filenames = os.listdir(ful_path)
            filenames.sort()
            for filename in filenames:
                img_path = os.path.join(ful_path, filename)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img_rgb is not None:
                    image_list.append(img_rgb)
                else:
                    print(f"Failed to read image: {img_path}")

            if image_list:
                return np.stack(image_list)
            else:
                return None

        # get frames:
        frames = read_and_stack_images(ful_path=video_path)


        resized_frames = []
        for i in range(frames.shape[0]):
            frame = np.array(
                Image.fromarray(frames[i]).convert('RGB').resize([self.sample_size[1], self.sample_size[0]]))
            resized_frames.append(frame)
        resized_frames = np.array(resized_frames)

        resized_frames = torch.tensor(resized_frames).permute(0, 3, 1, 2).float()  # [t,h,w,c] -> [t,c,h,w]

        crops = torch.tensor(ast.literal_eval(sample["crops"])[0]) if sample["crops"] != "[]" else torch.tensor([0,0,0,0])
        knife = torch.tensor(ast.literal_eval(sample["knife"])[0]) if sample["knife"] != "[]" else torch.tensor([0,0,0,0])
        mucosa = torch.tensor(ast.literal_eval(sample["mucosa"])[0]) if sample["mucosa"] != "[]" else torch.tensor([0,0,0,0])
        background = torch.tensor([0,0,1,0.77])
        bbox = torch.vstack([background,knife,mucosa,crops])*self.sample_size[1]
        return resized_frames, rel_path, bbox

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        pixel_values, video_name, bbox = self.get_batch(idx)

        
        pixel_values = pixel_values / 255.

        sample = dict(pixel_values=pixel_values, video_name=video_name, bbox=bbox)
        return sample