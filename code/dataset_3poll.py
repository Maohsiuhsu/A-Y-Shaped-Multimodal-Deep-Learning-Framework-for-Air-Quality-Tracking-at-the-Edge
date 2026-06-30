import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm  import tqdm
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset
import random
import xarray as xr
import rioxarray
from PIL import Image
class NO2PredictionDataset(Dataset):

    def __init__(self, datadir, samples, transforms=None, station_imgs=None):
        self.datadir = datadir
        self.transforms = transforms
        self.station_imgs = station_imgs # dict of AirQualityStation -> S2 image
        self.samples = samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.station_imgs is not None:
            sample["img"] = self.station_imgs.get(sample["AirQualityStation"])
            
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.samples)

    def display_sample(self, sample, title=None):
        img = sample["img"]
        band_data = self._normalize_for_display(img)
        fig, axs = plt.subplots(1, 2, figsize=(7,7))
        s2_ax = axs[0]
        s2_ax.imshow(band_data[:, :, [3,2,1]])
        s2_ax.set_title("Sentinel2 data")

        im = axs[1].imshow(sample["s5p"])
        axs[1].set_title("Sentinel-5P data")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        if title is not None:
            fig.suptitle(title)

        plt.show()

    def _normalize_for_display(self, band_data):
        band_data = reshape_as_image(np.array(band_data))
        lower_perc = np.percentile(band_data, 2, axis=(0,1))
        upper_perc = np.percentile(band_data, 98, axis=(0,1))
        return (band_data - lower_perc) / (upper_perc - lower_perc)
    

class PredictionDataset(Dataset):
    def __init__(self, datadir, samples, transforms=None):
        self.datadir = datadir
        self.samples = samples
        self.transforms = transforms

    def __getitem__(self, idx):
        # 1. 複製一份 sample，避免污染原始清單
        sample = self.samples[idx].copy()
        
        # 2. 組合影像的完整路徑
        image_name = sample['image_name']
        img_path = os.path.join(self.datadir, image_name)

        # 3. 嘗試讀取影像，加上終極防呆機制
        try:
            # 🌟 使用 PIL 讀取圖片，確保轉換為 RGB (避免黑白圖片少一個通道)，然後轉成 numpy 陣列
            with Image.open(img_path) as img:
                sample["img"] = np.array(img.convert("RGB"))
        except Exception as e:
            print(f"\n[警告] 訓練中突發無法讀取影像 {image_name}，自動隨機抽換其他樣本。錯誤訊息: {e}")
            random_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_idx)

        # 4. 做影像擴增與轉換
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.samples)

   

    # def display_sample(self, sample, title=None):
    #     img = sample["img"]
    #     band_data = self._normalize_for_display(img)
    #     fig, axs = plt.subplots(1, 2, figsize=(7,7))
    #     s2_ax = axs[0]
    #     s2_ax.imshow(band_data[:, :, [3,2,1]])
    #     s2_ax.set_title("Sentinel2 data")

    #     im = axs[1].imshow(sample["s5p"])
    #     axs[1].set_title("Sentinel-5P data")
    #     fig.subplots_adjust(right=0.8)
    #     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #     fig.colorbar(im, cax=cbar_ax)

    #     if title is not None:
    #         fig.suptitle(title)

    #     plt.show()

    # def _normalize_for_display(self, band_data):
    #     band_data = reshape_as_image(np.array(band_data))
    #     lower_perc = np.percentile(band_data, 2, axis=(0,1))
    #     upper_perc = np.percentile(band_data, 98, axis=(0,1))
    #     return (band_data - lower_perc) / (upper_perc - lower_perc)