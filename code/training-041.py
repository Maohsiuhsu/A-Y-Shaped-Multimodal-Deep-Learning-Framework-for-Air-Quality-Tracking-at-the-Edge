import os
from typing import Optional
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import sys
import copy
import random
import argparse
from datetime import datetime
from distutils.util import strtobool

import mlflow
import numpy as np
if not hasattr(np, "float"):
    np.float = float
import pandas as pd
from tqdm  import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset_3poll import NO2PredictionDataset, PredictionDataset
from transforms_3poll import ChangeBandOrder, ToTensor, DatasetStatistics, Normalize, Randomize
from model_3poll import get_model
from utils_3poll import load_data, none_or_true, dotdict, set_seed, step
from train_utils_3poll import eval_metrics, split_samples, test, test_plotter

parser = argparse.ArgumentParser(description='train_multimodel')
# network = "resnet18" #alternatively: "resnet50","resnet18""mobilenetv3small""efficientnet_b0"
#network = "efficientnet_b0"
#python training-041.py --samples_file "/home/ubuntu5080/4T_2/041/AQNet/train_2-dataset-all-tok.csv" --npz_file "/home/ubuntu5080/4T_2/041/AQNet/train_dataset-ok.npz" --epochs 100 --batch_size 8
# block for complete training
parser.add_argument('--samples_file', default="data/multimodal/samples_multimodal_3polls.csv", type=str)
parser.add_argument('--img_dir', default="", type=str)
#parser.add_argument('--npz_file', default="/home/ubuntu5080/4T_2/041/AQNet/train_dataset.npz", type=str)
parser.add_argument('--result_dir', default="results", type=str)
parser.add_argument('--checkpoint', default=True, type=str)
parser.add_argument('--epochs', default=30, type=int) 
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--runs', default=1, type=int)
parser.add_argument('--tabular', default="True", type=str)


parser.add_argument('--ablation_mode', default="both", type=str, choices=["both", "image_only", "tabular_only"], help="消融實驗模式切換")
parser.add_argument('--tabular_model', default="tabnet", type=str, 
                    choices=["tabnet", "mlp", "resnet", "linear", "shallow_mlp"], help="切換不同的文本模型架構")
parser.add_argument('--network', default="efficientnet_b0", type=str, 
                    choices=["mobilenetv3small", "resnet18", "resnet50", "vit", "efficientnet_b0" , "simple_cnn","densenet121","mobilenet_v2"], help="切換影像主幹網路")



# training parameters
parser.add_argument('--early_stopping', default="True", type=str)
parser.add_argument('--weight_decay_lambda', default=0.0001, type=float)
parser.add_argument('--learning_rate', default=0.0001, type=float)

args = parser.parse_args()
bool_args = ["early_stopping", "tabular"]
config = dotdict({k : strtobool(v) if k in bool_args else v for k,v in vars(args).items()})

S5p_switch = True
tabular_switch = config.tabular
tab_label = "SatDataOnly"
if tabular_switch == True:
    tab_label = "SatAndTabData"

print(tab_label)

# set internal parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

assert config.checkpoint in [True, None]
if config.checkpoint is True:
    checkpoint_name = "ImageNet"
else: checkpoint_name = "FromScratch"

experiment = "_".join([datetime.today().strftime('%Y-%m-%d-%H:%M:%S'), checkpoint_name, config.ablation_mode])

# generate output folder and ID
#experiment_folder_name = f"{datetime.today().strftime('%Y%m%d_%H%M%S')}_{network}_{config.ablation_mode}_{config.epochs}epochs"
experiment_folder_name = f"{datetime.today().strftime('%Y%m%d_%H%M%S')}_{config.network}_{config.ablation_mode}_{config.epochs}epochs"
output_directory = os.path.join(config.result_dir, experiment_folder_name)
os.makedirs(output_directory, exist_ok=True)
experiment_id = mlflow.create_experiment(experiment)

# print config info to cmd
print(config.samples_file)
print(config.datadir if hasattr(config, 'datadir') else "No datadir specified")
print(config.checkpoint)
print(device)
print("Start loading samples...")

# load data and instantiate objects
#samples = load_data(config.npz_file, config.samples_file)
samples = load_data(config.img_dir, config.samples_file)
# 🌟 使用 SmoothL1Loss 對抗極端污染值
loss = nn.SmoothL1Loss() 
datastats = DatasetStatistics()
tf = transforms.Compose([ChangeBandOrder(), Normalize(datastats), Randomize(), ToTensor()])

# set up performances lists
performances_test = []
performances_val = []
performances_train = []

# MODEL TRAINING
for run in tqdm(range(1, config.runs+1), unit="run"):

    # fix a different seed for each run
    seed = run

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("samples_file", config.samples_file)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("result_dir", config.result_dir)
        mlflow.log_param("pretrained_checkpoint", config.checkpoint)
        mlflow.log_param("device", device)
        mlflow.log_param("early_stopping", config.early_stopping)
        mlflow.log_param("learning_rate", config.learning_rate)
        mlflow.log_param("run", run)
        mlflow.log_param("weight_decay", config.weight_decay_lambda)
        mlflow.log_param("epochs", config.epochs)
        mlflow.log_param("seed", seed)
        mlflow.log_param("ablation_mode", config.ablation_mode)

        # set the seed for this run
        set_seed(seed)

        # initialize dataloaders + model
        print("Initializing dataset")
        samples_train, samples_val, samples_test = split_samples(samples, 0.1, 0.1)

        # 這裡的 config.datadir 如果沒有在 arg 裡，可以直接傳入 None 或 ""，因為我們改用 npz 了
        #dataset_test = PredictionDataset("", samples_test, transforms=tf)
        #dataset_train = PredictionDataset("", samples_train, transforms=tf)
        #dataset_val = PredictionDataset("", samples_val, transforms=tf)
        # 原本第一個參數是 ""，現在必須把資料夾路徑傳進去
        dataset_test = PredictionDataset(config.img_dir, samples_test, transforms=tf)
        dataset_train = PredictionDataset(config.img_dir, samples_train, transforms=tf)
        dataset_val = PredictionDataset(config.img_dir, samples_val, transforms=tf)

        #dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, num_workers=12, shuffle=True, pin_memory=False)
        dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, num_workers=8, shuffle=True, pin_memory=False, drop_last=True)
        dataloader_test = DataLoader(dataset_test, batch_size=8, num_workers=8, shuffle=False, pin_memory=False)
        dataloader_val = DataLoader(dataset_val, batch_size=8, num_workers=8, shuffle=False, pin_memory=False)
        dataloader_train_for_testing = DataLoader(dataset_train, batch_size=8, num_workers=8, shuffle=False, pin_memory=False)
        
        # instantiate model
        print(f"Initializing model in mode: {config.ablation_mode}")
        model = get_model(device, config.network, tabular_switch, S5p_switch, config.checkpoint, ablation_mode=config.ablation_mode, tabular_model=config.tabular_model)
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        #optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay_lambda)
        optimizer = torch.optim.AdamW([
    {
        'params': model.backbone_S2.parameters(),
        'lr': 1e-5,
        'weight_decay': config.weight_decay_lambda
    },
    {
        'params': model.backbone_tabular.parameters(),
        'lr': 1e-3,
        'weight_decay': config.weight_decay_lambda
    },
    {
        'params': model.shared_head.parameters(),
        'lr': 1e-3,
        'weight_decay': config.weight_decay_lambda
    },
    {
        'params': model.head_pm25.parameters(),
        'lr': 1e-3,
        'weight_decay': config.weight_decay_lambda
    },
    {
        'params': model.head_pm10.parameters(),
        'lr': 1e-3,
        'weight_decay': config.weight_decay_lambda
    },
    {
        'params': model.log_vars,
        'lr': 1e-3,
        'weight_decay': 0.0
    }
])
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

        print("Dataset Lengths - Train, Val, Test: "+str([len(dataset_train),len(dataset_val),len(dataset_test)]))

        print("Start training")
        # train the model
        best_val_mse = float('inf') # 初始誤差設為無限大
        best_model_path = os.path.join(output_directory, f"best_model_run{run}.pt")
        epoch_loss_records = []
        for epoch in range(config.epochs):
            model.train()

            # set up blank lists
            loss_history = []
            loss_epoch_list = []
            
            loss_pm25_list = []
            loss_pm10_list = []
            log_var_pm25_list = []
            log_var_pm10_list = []
            weight_pm25_list = []
            weight_pm10_list = []

            r2_train_pm25_list, mae_train_pm25_list, mse_train_pm25_list = [], [], []
            r2_train_pm10_list, mae_train_pm10_list, mse_train_pm10_list = [], [], []

            train_pbar = tqdm(dataloader_train, desc=f"Epoch {epoch}/{config.epochs-1} Training", leave=False)
            
            for idx, sample in enumerate(train_pbar):
                
                img = sample["img"].float().to(device)

                Onehot1 = sample["rural"]
                Onehot2 = sample["suburban"]
                Onehot3 = sample["urban"]
                Onehot4 = sample["traffic"]
                Onehot5 = sample["industrial"]
                Onehot6 = sample["background"]

                # 🌟 數值壓縮：將龐大的數字除以常數，縮放到 0~5 左右的範圍
                norm_altitude = sample["Altitude"] / 1000.0
                norm_pop_density = sample["PopulationDensity"] / 10000.0

                #tabular = [norm_altitude, norm_pop_density, Onehot1, Onehot2, Onehot3, Onehot4, Onehot5, Onehot6]
                #tabular = torch.stack(tabular, dim=1).float().to(device)

                # 🌟 新增：氣象參數縮放
                norm_temp = sample["TEMP"].float() / 40.0         # 假設台灣高溫極限約 40 度
                norm_rh = sample["RH"].float() / 100.0            # 濕度轉為 0~1
                norm_ws = sample["wind_speed"].float() / 20.0     # 假設風速最大約 20 m/s

                # 🌟 新增：風向週期性編碼 (將角度轉為徑度 Radian，再取 Sin/Cos)
                wd_rad = sample["wind_direc"].float() * (torch.pi / 180.0)
                wd_sin = torch.sin(wd_rad)
                wd_cos = torch.cos(wd_rad)

                # 🌟 把所有 13 個特徵包起來 (順序要和測試時一模一樣)
                tabular = [
                    norm_altitude, norm_pop_density, 
                    Onehot1, Onehot2, Onehot3, Onehot4, Onehot5, Onehot6,
                    norm_temp, norm_rh, norm_ws, wd_sin, wd_cos
                ]
                tabular = torch.stack(tabular, dim=1).float().to(device)

                model_input = {"img": img, "tabular": tabular}

                y_pm25 = sample["pm2.5"].float()
                y_pm10 = sample["pm10"].float()
                y_samples = [y_pm25, y_pm10]

                # 🌟 執行訓練步驟，取得 Loss 與分離的指標
                #loss_batch, metric_pm25, metric_pm10 = step(model_input, y_samples, model, loss, optimizer)
                loss_batch, metric_pm25, metric_pm10, loss_info = step(model_input, y_samples, model, loss, optimizer)

                #loss_epoch_list.append(loss_batch.item())
                loss_epoch_list.append(loss_batch.item())

                loss_pm25_list.append(loss_info["loss_pm25"])
                loss_pm10_list.append(loss_info["loss_pm10"])

                log_var_pm25_list.append(loss_info["log_var_pm25"])
                log_var_pm10_list.append(loss_info["log_var_pm10"])

                weight_pm25_list.append(loss_info["weight_pm25"])
                weight_pm10_list.append(loss_info["weight_pm10"])

                r2_train_pm25_list.append(metric_pm25[0])
                mae_train_pm25_list.append(metric_pm25[1])
                mse_train_pm25_list.append(metric_pm25[2])

                r2_train_pm10_list.append(metric_pm10[0])
                mae_train_pm10_list.append(metric_pm10[1])
                mse_train_pm10_list.append(metric_pm10[2])

            # 🌟 Epoch 結算：計算平均值 (縮排修正至迴圈外)
            loss_epoch = np.array(loss_epoch_list).mean()
            
            loss_pm25_epoch = np.array(loss_pm25_list).mean()
            loss_pm10_epoch = np.array(loss_pm10_list).mean()

            log_var_pm25_epoch = np.array(log_var_pm25_list).mean()
            log_var_pm10_epoch = np.array(log_var_pm10_list).mean()

            weight_pm25_epoch = np.array(weight_pm25_list).mean()
            weight_pm10_epoch = np.array(weight_pm10_list).mean()
            
            r2_train_pm25 = np.array(r2_train_pm25_list).mean()
            mae_train_pm25 = np.array(mae_train_pm25_list).mean()
            mse_train_pm25 = np.array(mse_train_pm25_list).mean()

            r2_train_pm10 = np.array(r2_train_pm10_list).mean()
            mae_train_pm10 = np.array(mae_train_pm10_list).mean()
            mse_train_pm10 = np.array(mse_train_pm10_list).mean()

            scheduler.step()
            #scheduler.step(loss_epoch)
            torch.cuda.empty_cache()
            loss_history.append(loss_epoch)

            # validation
            #val_y, val_y_hat = test(model, dataloader_val, device, datastats)
            # validation
            val_y, val_y_hat, _, _ = test(model, dataloader_val, device, datastats)
            eval_val_pm25 = eval_metrics(val_y["pm2.5"], val_y_hat["pm2.5"])
            eval_val_pm10 = eval_metrics(val_y["pm10"], val_y_hat["pm10"])
            eval_val = {"pm2.5":eval_val_pm25, "pm10":eval_val_pm10}

            # ==========================================
            # 🌟 訓練進度與指標輸出 (完美排版)
            # ==========================================
            print(f"\n--- Epoch {epoch} ---")
            print(f" [Train] PM 2.5 | R2: {r2_train_pm25:>6.4f} | MAE: {mae_train_pm25:>6.4f} | MSE: {mse_train_pm25:>6.4f}")
            print(f" [Train] PM 10  | R2: {r2_train_pm10:>6.4f} | MAE: {mae_train_pm10:>6.4f} | MSE: {mse_train_pm10:>6.4f}")

            print(f" [Loss]  Total | {loss_epoch:>8.5f}")
            print(f" [Loss]  PM2.5 | raw: {loss_pm25_epoch:>8.5f} | log_var: {log_var_pm25_epoch:>8.5f} | weight: {weight_pm25_epoch:>8.5f}")
            print(f" [Loss]  PM10  | raw: {loss_pm10_epoch:>8.5f} | log_var: {log_var_pm10_epoch:>8.5f} | weight: {weight_pm10_epoch:>8.5f}")

            print(f"-----------------------------------------------------------------")
            print(f" [Val]   PM 2.5 | R2: {eval_val['pm2.5'][0]:>6.4f} | MAE: {eval_val['pm2.5'][1]:>6.4f} | MSE: {eval_val['pm2.5'][2]:>6.4f}")
            print(f" [Val]   PM 10  | R2: {eval_val['pm10'][0]:>6.4f} | MAE: {eval_val['pm10'][1]:>6.4f} | MSE: {eval_val['pm10'][2]:>6.4f}\n")
            
            current_val_mse = eval_val['pm2.5'][2] + eval_val['pm10'][2]
            
            # 如果現在的誤差比歷史紀錄還要低，就觸發備份！
            if current_val_mse < best_val_mse:
                best_val_mse = current_val_mse
                torch.save(model.state_dict(), best_model_path)
            epoch_loss_records.append({
                "Epoch": epoch,

                "Train_Total_Loss": loss_epoch,

                "Train_PM2.5_Raw_Loss": loss_pm25_epoch,
                "Train_PM10_Raw_Loss": loss_pm10_epoch,

                "LogVar_PM2.5": log_var_pm25_epoch,
                "LogVar_PM10": log_var_pm10_epoch,

                "Weight_PM2.5": weight_pm25_epoch,
                "Weight_PM10": weight_pm10_epoch,

                "Val_Total_MSE": current_val_mse,
                "Val_PM2.5_MSE": eval_val['pm2.5'][2],
                "Val_PM10_MSE": eval_val['pm10'][2]
})
            

            # 紀錄數值
            performances_val.append([run, epoch] + eval_val["pm2.5"] + eval_val["pm10"] )
            mlflow.log_metrics({
                "val_r2_epoch_pm2.5" : eval_val["pm2.5"][0], "val_mae_epoch_pm2.5" : eval_val["pm2.5"][1], "val_mse_epoch_pm2.5" : eval_val["pm2.5"][2],
                "val_r2_epoch_pm10": eval_val["pm10"][0], "val_mae_epoch_pm10": eval_val["pm10"][1], "val_mse_epoch_pm10": eval_val["pm10"][2],
                "train_total_loss": loss_epoch,"train_pm2.5_raw_loss": loss_pm25_epoch,"train_pm10_raw_loss": loss_pm10_epoch,
                "log_var_pm2.5": log_var_pm25_epoch,"log_var_pm10": log_var_pm10_epoch,
                "weight_pm2.5": weight_pm25_epoch,"weight_pm10": weight_pm10_epoch,
            }, step=epoch)
            mlflow.log_metric("current_epoch", epoch, step=epoch)

        # ==========================================
        # 🌟 最終測試與大結算
        # ==========================================
        loss_df = pd.DataFrame(epoch_loss_records)
        loss_csv_path = os.path.join(output_directory, f"Epoch_Loss_History_run{run}.csv")
        loss_df.to_csv(loss_csv_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 訓練過程的 Epoch Loss 歷史已成功存檔至: {loss_csv_path}")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # 接收測試結果與照片檔名，以及特徵遮罩
        test_y, test_y_hat, test_img_names, test_feature_masks = test(model, dataloader_test, device, datastats)
        train_y, train_y_hat, _, _ = test(model, dataloader_train_for_testing, device, datastats)

        # ==========================================
        # Output prediction details & Feature Masks to CSV
        # ==========================================
        
        # 1. 準備基礎的 CSV 欄位資料字典 (就是這裡遺失了！)
        csv_data = {
            "Image Name": test_img_names,
            "PM2.5 μg/m³(True Value)": test_y["pm2.5"],
            "PM10 μg/m³(True Value)": test_y["pm10"],
            "PM2.5 μg/m³(Predicted Value)": test_y_hat["pm2.5"],
            "PM10 μg/m³(Predicted Value)": test_y_hat["pm10"]
        }

        # 2. 如果有成功收集到特徵遮罩，將其拆解成 13 個欄位塞進字典
        if len(test_feature_masks) > 0:
            feature_names = [
                "Weight_Altitude", "Weight_PopDensity", 
                "Weight_Rural", "Weight_Suburban", "Weight_Urban", 
                "Weight_Traffic", "Weight_Industrial", "Weight_Background",
                "Weight_Temp", "Weight_RH", "Weight_WindSpeed", 
                "Weight_WindDir_Sin", "Weight_WindDir_Cos"
            ]
            
            masks_array = np.array(test_feature_masks)
            
            for i, name in enumerate(feature_names):
                csv_data[name] = masks_array[:, i]

        # 3. 將字典正式轉換為 DataFrame 並存檔
        test_details_df = pd.DataFrame(csv_data)
        
        csv_filename = os.path.join(output_directory, "Test_Predictions_Detail.csv")
        test_details_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
        print(f"\n✅ Prediction details and Feature Importances saved to: {csv_filename}")
        
        # ==========================================
        # (這裡下面繼續接你原本計算 eval_test_pm25, eval_train_pm25 的程式碼...)

        
        
        eval_test_pm25 = eval_metrics(test_y["pm2.5"], test_y_hat["pm2.5"])
        eval_test_pm10 = eval_metrics(test_y["pm10"], test_y_hat["pm10"])
        eval_test = {"pm2.5": eval_test_pm25, "pm10": eval_test_pm10}

        eval_train_pm25 = eval_metrics(train_y["pm2.5"], train_y_hat["pm2.5"])
        eval_train_pm10 = eval_metrics(train_y["pm10"], train_y_hat["pm10"])
        eval_train = {"pm2.5": eval_train_pm25, "pm10": eval_train_pm10}

        print(f" [Train] PM 2.5 | R2: {eval_train['pm2.5'][0]:>6.4f} | MAE: {eval_train['pm2.5'][1]:>6.4f} | MSE: {eval_train['pm2.5'][2]:>6.4f}")
        print(f" [Train] PM 10  | R2: {eval_train['pm10'][0]:>6.4f} | MAE: {eval_train['pm10'][1]:>6.4f} | MSE: {eval_train['pm10'][2]:>6.4f}")
     
        print(f" [Test]  PM 2.5 | R2: {eval_test['pm2.5'][0]:>6.4f} | MAE: {eval_test['pm2.5'][1]:>6.4f} | MSE: {eval_test['pm2.5'][2]:>6.4f}")
        print(f" [Test]  PM 10  | R2: {eval_test['pm10'][0]:>6.4f} | MAE: {eval_test['pm10'][1]:>6.4f} | MSE: {eval_test['pm10'][2]:>6.4f}\n")

        test_plotter(output_directory, test_y, test_y_hat, train_y, train_y_hat)

        mlflow.log_metric("test_r2_pm2.5", eval_test["pm2.5"][0])
        mlflow.log_metric("test_mae_pm2.5", eval_test["pm2.5"][1])
        mlflow.log_metric("test_mse_pm2.5", eval_test["pm2.5"][2])

        # 🌟 新增 2：把實驗模式、網路名稱、參數量一起存進測試紀錄中
        performances_test.append([
            config.ablation_mode,    # 紀錄是 both, image_only 還是 tabular_only
            config.network,          # 紀錄影像模型 (例如 efficientnet_b0)
            config.tabular_model,    # 紀錄文本模型 (例如 tabnet)
            total_params,            # 🌟 紀錄總參數量
            eval_test["pm2.5"][0], eval_test["pm2.5"][1], eval_test["pm2.5"][2],
            eval_test["pm10"][0], eval_test["pm10"][1], eval_test["pm10"][2]
        ])
        
        performances_train.append([eval_train["pm2.5"][0], eval_train["pm2.5"][1], eval_train["pm2.5"][2],
                                  eval_train["pm10"][0], eval_train["pm10"][1], eval_train["pm10"][2]])

        mlflow.log_artifacts(output_directory) 

# set up performance dfs
performances_val = pd.DataFrame(performances_val, columns=["run", "epoch", "r2_pm2.5", "mae_pm2.5", "mse_pm2.5", "r2_pm10", "mae_pm10", "mse_pm10"])
# 🌟 新增 3：更新 CSV 的欄位名稱
performances_test = pd.DataFrame(performances_test, columns=[
    "Ablation_Mode", "Image_Model", "Tabular_Model", "Total_Params", 
    "r2_pm2.5", "mae_pm2.5", "mse_pm2.5", "r2_pm10", "mae_pm10", "mse_pm10"
])
performances_train = pd.DataFrame(performances_train,columns=["r2_pm2.5", "mae_pm2.5", "mse_pm2.5", "r2_pm10", "mae_pm10", "mse_pm10"])

# save results
print("Writing results...")
performances_test.to_csv(os.path.join(output_directory, "_".join([str(checkpoint_name), "test", str(config.epochs), "epochs"]) + ".csv"), index=False)
performances_train.to_csv(os.path.join(output_directory, "_".join([str(checkpoint_name), "train", str(config.epochs), "epochs"]) + ".csv"), index=False)
performances_val.to_csv(os.path.join(output_directory, "_".join([str(checkpoint_name), "val", str(config.epochs), "epochs"]) + ".csv"), index=False)

# save the model
print("Writing model...")
torch.save(model.state_dict(), os.path.join(output_directory, "_".join([str(checkpoint_name), str(config.epochs), "epochs"]) + ".model"))
print("done.")