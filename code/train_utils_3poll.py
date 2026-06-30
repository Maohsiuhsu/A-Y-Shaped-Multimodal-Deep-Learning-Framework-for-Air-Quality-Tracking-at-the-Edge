import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transforms_3poll import Normalize

def eval_metrics(y, y_hat):
    r2 = r2_score(y, y_hat)
    mae = mean_absolute_error(y, y_hat)
    mse = mean_squared_error(y, y_hat)
    return [r2, mae, mse]

def split_samples(samples, test_size=0.2, val_size=0.2, random_state=42):
    train_val, samples_test = train_test_split(samples, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1 - test_size)  
    samples_train, samples_val = train_test_split(train_val, test_size=val_ratio, random_state=random_state)
    return samples_train, samples_val, samples_test

def split_samples_df(samples, test_size=0.2, val_size=0.2):
    stations = samples.AirQualityStation.unique()
    stations_train, stations_test = train_test_split(stations, test_size=test_size)
    real_val_size = val_size / (1 - test_size)
    stations_train, stations_val = map(set, train_test_split(stations_train, test_size=real_val_size))
    stations_test = set(stations_test)

    samples_train = samples[samples.AirQualityStation.isin(stations_train)]
    samples_val = samples[samples.AirQualityStation.isin(stations_val)]
    samples_test = samples[samples.AirQualityStation.isin(stations_test)]

    return samples_train, samples_val, samples_test

def test(model, dataloader, device, datastats):
    model.eval()

    measurements_pm25 =[]
    measurements_pm10 =[]
    predictions_pm25 =[]
    predictions_pm10 =[]
    image_names = [] 
    feature_masks = [] # 🌟 新增：收集遮罩

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataloader, desc="Final testing and evaluation are underway.")):
            img = sample["img"].float().to(device)
            
            # ==========================================
            # 🌟 這裡是你遺失的 tabular 組合區塊 (13 個特徵)
            # ==========================================
            norm_altitude = sample["Altitude"] / 1000.0
            norm_pop_density = sample["PopulationDensity"] / 10000.0

            norm_temp = sample["TEMP"].float() / 40.0
            norm_rh = sample["RH"].float() / 100.0
            norm_ws = sample["wind_speed"].float() / 20.0

            wd_rad = sample["wind_direc"].float() * (torch.pi / 180.0)
            wd_sin = torch.sin(wd_rad)
            wd_cos = torch.cos(wd_rad)

            tabular = [
                norm_altitude, norm_pop_density,
                sample["rural"], sample["suburban"], sample["urban"],
                sample["traffic"], sample["industrial"], sample["background"],
                norm_temp, norm_rh, norm_ws, wd_sin, wd_cos
            ]
            tabular = torch.stack(tabular, dim=1).float().to(device)
            # ==========================================

            model_input = {"img": img, "tabular": tabular}

            y_hat_pm25, y_hat_pm10 = model(model_input)
            
            y_pm25 = sample["pm2.5"].float().to(device)
            y_pm10 = sample["pm10"].float().to(device)

            measurements_pm25.extend(y_pm25.cpu().numpy().flatten().tolist())
            measurements_pm10.extend(y_pm10.cpu().numpy().flatten().tolist())

            predictions_pm25.extend(y_hat_pm25.cpu().detach().numpy().flatten().tolist())
            predictions_pm10.extend(y_hat_pm10.cpu().detach().numpy().flatten().tolist())
            
            image_names.extend(sample["image_name"])

            # 🌟 嘗試取得 TabNet 的特徵遮罩
            if hasattr(model, 'backbone_tabular') and hasattr(model.backbone_tabular, 'current_feature_mask'):
                batch_masks = model.backbone_tabular.current_feature_mask.cpu().numpy()
                feature_masks.extend(batch_masks.tolist())

    measurements_pm25 = Normalize.undo_pm25_standardization(datastats, np.array(measurements_pm25))
    measurements_pm10 = Normalize.undo_pm10_standardization(datastats, np.array(measurements_pm10))
    predictions_pm25 = Normalize.undo_pm25_standardization(datastats, np.array(predictions_pm25))
    predictions_pm10 = Normalize.undo_pm10_standardization(datastats, np.array(predictions_pm10))

    measurements = {"pm2.5":measurements_pm25, "pm10":measurements_pm10}
    predictions = {"pm2.5":predictions_pm25, "pm10":predictions_pm10}

    # 🌟 回傳 4 個變數
    return measurements, predictions, image_names, feature_masks

def test_plotter(output_directory,test_y, test_y_hat, train_y, train_y_hat):
    # 改為 2x2 的圖表矩陣，完美對應 4 張圖表
    img, axs = plt.subplots(2, 2, figsize=(12, 10))
    img.subplots_adjust(hspace=0.4, wspace=0.3)

    # === [左上] PM2.5 Test ===
    axs[0, 0].scatter(test_y["pm2.5"], test_y_hat["pm2.5"], s=2)
    axs[0, 0].set_title("PM2.5 test")
    axs[0, 0].set_xlabel("True Value")
    axs[0, 0].set_ylabel("Predicted Value")
    axs[0, 0].axline((0, 0), slope=1, c="red")
    
    # === [右上] PM2.5 Train ===
    axs[0, 1].scatter(train_y["pm2.5"], train_y_hat["pm2.5"], s=2)
    axs[0, 1].set_title("PM2.5 train")
    axs[0, 1].set_xlabel("True Value")
    axs[0, 1].set_ylabel("Predicted Value")
    axs[0, 1].axline((0, 0), slope=1, c="red")

    # === [左下] PM10 Test ===
    axs[1, 0].scatter(test_y["pm10"], test_y_hat["pm10"], s=2)
    axs[1, 0].set_title("PM10 test")
    axs[1, 0].set_xlabel("True Value")
    axs[1, 0].set_ylabel("Predicted Value")
    axs[1, 0].axline((0, 0), slope=1, c="red")
    
    # === [右下] PM10 Train ===
    axs[1, 1].scatter(train_y["pm10"], train_y_hat["pm10"], s=2)
    axs[1, 1].set_title("PM10 train")
    axs[1, 1].set_xlabel("True Value")
    axs[1, 1].set_ylabel("Predicted Value")
    axs[1, 1].axline((0, 0), slope=1, c="red")

    plt.savefig(output_directory + "/predictions.png")
    plt.close()