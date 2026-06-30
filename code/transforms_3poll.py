import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from rasterio.plot import reshape_as_image

# define image transforms
# class ChangeBandOrder(object):
#     def __call__(self, sample):
#         img = sample["img"].copy()
#         img = np.moveaxis(img, -1, 0)
        
#         # 確保影像一定是 960x960 (使用 PyTorch 的 Interpolate 來強制縮放，取代原本危險的裁切)
#         if img.shape[1] != 1080 or img.shape[2] != 1920:
#             # 將 numpy 轉為 tensor，並加上 Batch 維度 (1, C, H, W)
#             tensor_img = torch.from_numpy(img).unsqueeze(0).float()
#             # 強制縮放為 960x960
#             tensor_img = F.interpolate(tensor_img, size=(1080, 1920), mode='bilinear', align_corners=False)
#             # 轉回 numpy 陣列
#             reordered_img = tensor_img.squeeze(0).numpy()
#         else:
#             reordered_img = img

#         out = {}
#         for k,v in sample.items():
#             if k == "img":
#                 out[k] = reordered_img
#             else:
#                 out[k] = v

#         return out

class ChangeBandOrder(object):
    def __call__(self, sample):
        img = sample["img"].copy()
        
        # ==========================================
        # 🌟 自動塗黑浮水印機制：
        # 根據你小畫家的測試，我們把左上角那一小塊塗成全黑
        # 假設高度蓋到 100，寬度蓋到 600 (你可以根據實際需要微調數字)
        # ==========================================
        img[0:70, 0:750, :] = 0 
        
        img = np.moveaxis(img, -1, 0)
        
        # 確保影像一定是 1080x1920
        if img.shape[1] != 1080 or img.shape[2] != 1920:
            tensor_img = torch.from_numpy(img).unsqueeze(0).float()
            tensor_img = F.interpolate(tensor_img, size=(1080, 1920), mode='bilinear', align_corners=False)
            reordered_img = tensor_img.squeeze(0).numpy()
        else:
            reordered_img = img

        out = {}
        for k,v in sample.items():
            if k == "img":
                out[k] = reordered_img
            else:
                out[k] = v

        return out
    
    
class ToTensor(object):
    def __call__(self, sample):
        img = torch.from_numpy(sample["img"].copy())

        # if sample.get("so2") is not None:
        #     so2 = torch.from_numpy(sample["so2"].copy())
        # if sample.get("no") is not None:
        #     no = torch.from_numpy(sample["no"].copy())
        # if sample.get("co") is not None:
        #     co = torch.from_numpy(sample["co"].copy())
        # if sample.get("o3") is not None:
        #     o3 = torch.from_numpy(sample["o3"].copy())
        # if sample.get("pm2.5") is not None:
        #     pm25 = torch.from_numpy(sample["pm2.5"].copy())
        # if sample.get("pm10") is not None:
        #     pm10 = torch.from_numpy(sample["pm10"].copy())
        # if sample.get("s5p") is not None:
        #     s5p = torch.from_numpy(sample["s5p"].copy())
        if sample.get("so2") is not None:
            so2 = torch.tensor(sample["so2"], dtype=torch.float32)  # 直接轉換成 tensor
        if sample.get("no") is not None:
            no = torch.tensor(sample["no"], dtype=torch.float32)
        if sample.get("co") is not None:
            co = torch.tensor(sample["co"], dtype=torch.float32)
        if sample.get("o3") is not None:
            o3 = torch.tensor(sample["o3"], dtype=torch.float32)
        if sample.get("pm2.5") is not None:
            pm25 = torch.tensor(sample["pm2.5"], dtype=torch.float32)
        if sample.get("pm10") is not None:
            pm10 = torch.tensor(sample["pm10"], dtype=torch.float32)

        out = {}
        for k,v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "so2":
                out[k] = so2
            elif k == "no":
                out[k] = no
            elif k == "co":
                out[k] = co
            elif k == "o3":
                out[k] = o3
            elif k == "pm10":
                out[k] = pm10
            elif k == "pm2.5":
                out[k] = pm25
            else:
                out[k] = v

        return out


class DatasetStatistics(object):
    def __init__(self):
        
        self.channel_means = np.array([0.5, 0.5, 0.5])
        self.channel_std = np.array([0.5, 0.5, 0.5])

        self.so2_mean = 1.523125
        self.so2_std = 0.4155035716875444

        self.no_mean = 1.8229166666666667
        self.no_std = 1.078424694957898

        self.co_mean = 0.32916666666666666
        self.co_std = 0.08960512270057107

        self.o3_mean = 38.262708333333336
        self.o3_std = 14.186758695914273

        #self.pm25_mean = 13.502134146341463 #0904
        #self.pm25_std = 3.7315943850762507
        #self.pm10_mean = 23.341463414634145
        #self.pm10_std = 5.159165739151988
        
        # self.pm25_mean = 10.040986394557823 #all
        # self.pm25_std = 4.981458911725346
        # self.pm10_mean = 18.96825396825397
        # self.pm10_std = 6.792952130796215
        
        
        # self.pm25_mean = 20.72   #0326-0330
        # self.pm25_std = 10.100500357046009
        # self.pm10_mean = 38.355223880597016
        # self.pm10_std = 16.520157597293974
        
        #self.pm25_mean = 16.257737063629612 #0326-0407
        #self.pm25_std = 9.0871096723745
        #self.pm10_mean = 29.51076999257242
        #self.pm10_std = 16.404874316442736
        

        self.pm25_mean = 15.888791746808883  #0326-0415
        self.pm25_std = 8.037171810630126
        self.pm10_mean = 28.59905577898234
        self.pm10_std = 14.523242521376316


class Normalize(object):
    """normalize a sample, i.e. the image and NO2 value, by subtracting mean and dividing by std"""
    def __init__(self, statistics):
        self.statistics = statistics

    def __call__(self, sample):
        img = reshape_as_image(sample.get("img").copy())
        img = np.moveaxis((img - self.statistics.channel_means) / self.statistics.channel_std, -1, 0)

        if sample.get("so2") is not None:
            so2 = sample.get("so2")#.copy()
            so2 = np.array((so2 - self.statistics.so2_mean) / self.statistics.so2_std)
            #no2 = np.array((no2 - 0) / 1)

        if sample.get("no") is not None:
            no = sample.get("no")#.copy()
            no = np.array((no - self.statistics.no_mean) / self.statistics.no_std)

        if sample.get("co") is not None:
            co = sample.get("co")#.copy()
            co = np.array((co - self.statistics.co_mean) / self.statistics.co_std)
        
        if sample.get("o3") is not None:
            o3 = sample.get("o3")#.copy()
            o3 = np.array((o3 - self.statistics.o3_mean) / self.statistics.o3_std)

        if sample.get("pm2.5") is not None:
            pm25 = sample.get("pm2.5")#.copy()
            pm25 = np.array((pm25 - self.statistics.pm25_mean) / self.statistics.pm25_std)

        if sample.get("pm10") is not None:
            pm10 = sample.get("pm10")#.copy()
            pm10 = np.array((pm10 - self.statistics.pm10_mean) / self.statistics.pm10_std)
        
        out = {}
        for k,v in sample.items():
            if k == "img":
                out[k] = img
            elif k == "so2":
                out[k] = so2
            elif k == "no":
                out[k] = no
            elif k == "co":
                out[k] = co
            elif k == "o3":
                out[k] = o3
            elif k == "pm2.5":
                out[k] = pm25
            elif k == "pm10":
                out[k] = pm10
            else:
                out[k] = v

        return out
    @staticmethod
    def undo_so2_standardization(statistics, so2):
        return (so2 * statistics.so2_std) + statistics.so2_mean
    @staticmethod
    def undo_no_standardization(statistics, no):
        return (no * statistics.no_std) + statistics.no_mean
    @staticmethod
    def undo_co_standardization(statistics, co):
        return (co * statistics.co_std) + statistics.co_mean
    @staticmethod
    def undo_o3_standardization(statistics, o3):
        return (o3 * statistics.o3_std) + statistics.o3_mean
    @staticmethod
    def undo_pm25_standardization(statistics, pm25):
        return (pm25 * statistics.pm25_std) + statistics.pm25_mean
    @staticmethod
    def undo_pm10_standardization(statistics, pm10):
        return (pm10 * statistics.pm10_std) + statistics.pm10_mean

class Randomize():
    def __call__(self, sample):
        img = sample.get("img").copy()

        s5p_available = False
        

        if random.random() > 0.5:
            img = np.flip(img, 1)
            if s5p_available: s5p = np.flip(s5p, 0)
        if random.random() > 0.5:
            img = np.flip(img, 2)
            if s5p_available: s5p = np.flip(s5p, 1)
        #if random.random() > 0.5:
        #    img = np.rot90(img, np.random.randint(0, 4), axes=(1,2))
        #    if s5p_available: s5p = np.rot90(s5p, np.random.randint(0, 4), axes=(0,1))

        out = {}
        for k,v in sample.items():
            if k == "img":
                out[k] = img
            else:
                out[k] = v

        return out