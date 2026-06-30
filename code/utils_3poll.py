import os
from re import S
import numpy as np
import pandas as pd
from tqdm  import tqdm
import matplotlib.pyplot as plt
from rasterio.plot import reshape_as_image
from torch.utils.data import Dataset
from PIL import Image
import torch
import random

import xarray as xr
import rioxarray

from train_utils_3poll import eval_metrics

def read_param_file(filepath):
    with open(filepath, "r") as f:
        output = f.read()
    return output

# def step(x, y_samples, model, loss, optimizer):

#     y_no2, y_o3, y_pm10 = y_samples
#     y_hat_1,y_hat_2,y_hat_3 = model(x)

#     y_train = torch.stack([y_no2, y_o3, y_pm10])
#     y_epoch = torch.stack([y_hat_1,y_hat_2,y_hat_3])
#     loss_epoch = loss(y_epoch,y_train.to("cuda:0"))

#     optimizer.zero_grad()
#     loss_epoch.backward()
#     optimizer.step()

#     metric_results = eval_metrics(y_train.detach().cpu(),y_epoch.detach().cpu())
#     #print(metric_results)

#     return loss_epoch.detach().cpu(), metric_results


def step(x, y_samples, model, loss_fn, optimizer):

    y_pm25, y_pm10 = y_samples

    # ✅ 修改 1：不要硬寫 cuda:0，改成跟著 model 的 device
    device = next(model.parameters()).device
    y_pm25 = y_pm25.to(device)
    y_pm10 = y_pm10.to(device)

    y_hat_pm25, y_hat_pm10 = model(x)

    # 1. 分別計算 PM2.5 與 PM10 的原始 loss
    loss_pm25 = loss_fn(y_hat_pm25, y_pm25)
    loss_pm10 = loss_fn(y_hat_pm10, y_pm10)

    # 2. 使用 uncertainty weighting 動態調整兩個任務的權重
    precision_pm25 = torch.exp(-model.log_vars[0])
    loss_1 = precision_pm25 * loss_pm25 + model.log_vars[0]

    precision_pm10 = torch.exp(-model.log_vars[1])
    loss_2 = precision_pm10 * loss_pm10 + model.log_vars[1]

    loss_epoch = loss_1 + loss_2

    # 3. 反向傳播與優化
    optimizer.zero_grad()
    loss_epoch.backward()
    optimizer.step()

    # 4. 分別計算 PM2.5 / PM10 的訓練指標
    metric_pm25 = eval_metrics(y_pm25.detach().cpu(), y_hat_pm25.detach().cpu())
    metric_pm10 = eval_metrics(y_pm10.detach().cpu(), y_hat_pm10.detach().cpu())

    # ✅ 修改 2：額外回傳 loss 與 log_vars 資訊，方便 training-041.py 紀錄
    with torch.no_grad():
        loss_info = {
            "loss_pm25": loss_pm25.detach().cpu().item(),
            "loss_pm10": loss_pm10.detach().cpu().item(),

            "log_var_pm25": model.log_vars[0].detach().cpu().item(),
            "log_var_pm10": model.log_vars[1].detach().cpu().item(),

            "weight_pm25": torch.exp(-model.log_vars[0]).detach().cpu().item(),
            "weight_pm10": torch.exp(-model.log_vars[1]).detach().cpu().item(),
        }

    return loss_epoch.detach().cpu(), metric_pm25, metric_pm10, loss_info


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# def load_data(datadir, samples_file):
#     """load samples to memory, returns array of samples and array of stations
#     each sample is a dict
#     this version loads all samples from one station in one go (e.g. for multiple months), s.t. the S5P data for the station is only read once"""

#     if not isinstance(samples_file, pd.DataFrame):
#         samples_df = pd.read_csv(samples_file, index_col="idx")
#     else:
#         samples_df = samples_file
#     samples_df = samples_df[np.isnan(samples_df.no2) == False]

#     print("Available columns from samples_file :")
#     print(samples_df.columns)

#     samples = []
#     stations = {}
#     try:
#         # here we assume that all S5P data for one station is stored in one .netcdf file
#         # so it's faster to access the samples on a per station basis and only opening the
#         # .netcdf file once
#         for station in tqdm(samples_df.AirQualityStation.unique()):
#             station_obs = samples_df[samples_df.AirQualityStation == station]
#             s5p_path = station_obs.s5p_path.unique().item()
#             s5p_data = xr.open_dataset(os.path.join(datadir, "sentinel-5p", s5p_path)).rio.write_crs(4326)

#             for idx in station_obs.index.values:
#                 sample = samples_df.loc[idx].to_dict() # select by index value, not position
#                 sample["idx"] = idx
#                 sample["s5p"] = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()
#                 samples.append(sample)
#                 stations[sample["AirQualityStation"]] = np.load(os.path.join(datadir, "sentinel-2", sample["img_path"]))

#             s5p_data.close()

#     except IndexError as e:
#         print(e)
#         print("idx:", idx)
#         print()

#     #print(samples)
#     return samples, stations

def load_data(img_dir, data_file):
    data_df = pd.read_csv(data_file)
    print("Available columns from data_file :")
    print(data_df.columns.tolist())
    weather_cols = ['TEMP', 'RH', 'wind_speed', 'wind_direc']
    for col in weather_cols:
        if col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce').fillna(0.0)
    samples = []
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"[錯誤] 找不到影像資料夾：{img_dir}\n請確認路徑是否正確！")
        
    print(f"正在建立資料清單 (嚴格檢查：確認實體檔案是否存在且可讀取)...")
    
    skipped_missing = 0
    skipped_corrupted = 0

    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="配對與過濾標籤"):
        sample = row.to_dict()
        image_name = sample['image_name']
        
        # 💡 注意：如果你的 csv 裡面的檔名沒有包含副檔名(如 .npy)，請在這裡補上
        # 例如： img_path = os.path.join(img_dir, image_name + ".npy")
        img_path = os.path.join(img_dir, image_name)
        
        # 1. 檢查檔案是否遺失 (沒有配對到)
        if not os.path.exists(img_path):
            skipped_missing += 1
            continue
            
        # 2. 檢查檔案是否損毀 (不能讀取)
        try:
            # 🌟 改用 PIL 來檢查 JPG 圖片
            with Image.open(img_path) as img:
                img.verify() # verify() 是一個超快的方法，只檢查圖片檔案結構是否損壞，不載入記憶體
            samples.append(sample)
        except Exception as e:
            skipped_corrupted += 1
            if skipped_corrupted <= 3:
                print(f"\n[檢查發現壞檔] {image_name} 無法讀取，錯誤原因：{e}")
    
    
    
    
    
    return samples

def none_or_true(value):
    if value == 'None':
        return None
    elif value == "True":
        return True
    return value

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__