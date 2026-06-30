import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, resnet18, resnet50, vit_b_16, efficientnet_b0,mobilenet_v2, densenet121


class Linear_TabularBackbone(nn.Module):
    """極致基礎 1：純線性映射 (等同於多元線性迴歸)"""
    def __init__(self, input_dim=13, output_dim=64):
        super(Linear_TabularBackbone, self).__init__()
        # 只有單純的權重矩陣相乘，沒有任何 ReLU 啟動函數
        self.network = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.network(x)
    
class Shallow_MLP_TabularBackbone(nn.Module):
    """極致基礎 2：單隱藏層淺層神經網路"""
    def __init__(self, input_dim=13, output_dim=64):
        super(Shallow_MLP_TabularBackbone, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)



class MLP_TabularBackbone(nn.Module):
    """基準模型：純粹的全連接層"""
    def __init__(self, input_dim=13, output_dim=64):
        super(MLP_TabularBackbone, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.network(x)

class ResNet_TabularBackbone(nn.Module):
    """進階模型：帶有殘差連接的表格網路"""
    def __init__(self, input_dim=13, output_dim=64):
        super(ResNet_TabularBackbone, self).__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        self.final_layer = nn.Linear(64, output_dim)
    def forward(self, x):
        identity = self.input_layer(x)
        out = self.block1(identity)
        return self.final_layer(out + identity) # 殘差相加



#表格使用了TabNet (Attentive Interpretable Tabular Learning)
class GLU_Layer(nn.Module):
    """Gated Linear Unit (門控線性單元)：控制資訊流動的開關"""
    def __init__(self, input_dim, output_dim):
        super(GLU_Layer, self).__init__()
        # 輸出維度設為兩倍，因為一半要做為特徵，另一半要做為「門控開關」
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        x = self.fc(x)
        # 切割成兩半：前一半是提煉的特徵，後一半經過 Sigmoid 變成 0~1 的開關
        features = x[:, :x.shape[1]//2]
        gates = torch.sigmoid(x[:, x.shape[1]//2:])
        # 核心運算：特徵乘上開關
        return features * gates

class AttentiveFeatureSelection(nn.Module):
    """注意力特徵篩選器：產生 0~1 的遮罩，決定哪些天氣/地理特徵該被忽略"""
    def __init__(self, input_dim):
        super(AttentiveFeatureSelection, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # 計算每個特徵的原始分數
        mask_values = self.fc(x)
        # 透過 Softmax 將分數轉換為總和為 1 的權重分配 (注意力遮罩)
        mask = F.softmax(mask_values, dim=-1)
        # 為了保持特徵的數值量級，將 mask 乘上特徵數量 (13)
        return mask * mask.shape[-1]

class TabNetStyleBackbone(nn.Module):
    def __init__(self, input_dim=13, output_dim=64):
        super(TabNetStyleBackbone, self).__init__()
        
        # 1. 產生遮罩的模組
        self.attentive_transformer = AttentiveFeatureSelection(input_dim)
        
        # 2. 處理篩選後特徵的轉換器 (使用兩層 GLU)
        self.feature_transformer = nn.Sequential(
            GLU_Layer(input_dim, 64),
            nn.BatchNorm1d(64),
            GLU_Layer(64, output_dim)
        )

    def forward(self, x):
        # 步驟一：計算出 13 個特徵的重要性遮罩
        mask = self.attentive_transformer(x)
        
        # 步驟二：強制篩選！將原始特徵乘上遮罩 (不重要的特徵會直接歸零或變極小)
        masked_x = x * mask
        
        # 步驟三：將過濾後的純粹特徵送入 GLU 轉換器
        out = self.feature_transformer(masked_x)
        
        # 💡 將遮罩存放在模型物件中，未來可以用來畫「特徵重要性圖表」
        self.current_feature_mask = mask.detach()
        
        return out
    
    
    
class ViTWrapper(nn.Module):
    def __init__(self, vit_model):
        super(ViTWrapper, self).__init__()
        self.vit = vit_model

    def forward(self, x):
        # 強制將輸入的 (1080, 1920) 縮放成 ViT 規定的 (224, 224)
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.vit(x_resized)
    
    
class SimpleCNN_Backbone(nn.Module):
    """極致基礎影像模型：自建 3 層卷積網路"""
    def __init__(self, output_dim=640):
        super(SimpleCNN_Backbone, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 強制將特徵圖縮小到 4x4 大小，方便接全連接層
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
def get_model(device, network, tabular_switch, S5p_switch, checkpoint=None, ablation_mode="both", tabular_model="tabnet"):
    tabular_input_count = 13
    S2_num_features = 640
    tabular_features = 64

    # === 左腦：影像特徵萃取 (CNN/Transformer Backbone) ===
    assert network in ["mobilenetv3small", "resnet18", "resnet50", "vit", "efficientnet_b0"]
    if network == "mobilenetv3small":
        backbone_S2 = mobilenet_v3_small(pretrained=checkpoint, num_classes=1000)
        backbone_S2.features[0][0] = nn.Conv2d(3, 16, 3, 1, 1)
        backbone_S2.classifier[3] = nn.Linear(1024, S2_num_features)
        
    elif network == "resnet18":
        backbone_S2 = resnet18(pretrained=checkpoint, num_classes=1000)
        backbone_S2.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2),padding=(3, 3), bias=False)
        backbone_S2.fc = nn.Linear(512, S2_num_features)
        
    elif network == "mobilenet_v2":
        print("[INFO] 載入邊緣運算經典基準 MobileNet-V2 影像主幹...")
        backbone_S2 = mobilenet_v2(pretrained=checkpoint)
        # MobileNetV2 的 classifier 第二層是輸出層
        backbone_S2.classifier[1] = nn.Linear(1280, S2_num_features)
        
    elif network == "densenet121":
        print("[INFO] 載入特徵重複利用經典 DenseNet-121 影像主幹...")
        backbone_S2 = densenet121(pretrained=checkpoint)
        # DenseNet 的分類頭就叫 classifier
        backbone_S2.classifier = nn.Linear(1024, S2_num_features)
        
    elif network == "simple_cnn":
        print("[INFO] 載入極簡 3 層 CNN 影像主幹 (無預訓練)...")
        # 這個是我們自建的，直接給它輸出維度 640 即可
        backbone_S2 = SimpleCNN_Backbone(output_dim=S2_num_features)    
        
    elif network == "resnet50":
        backbone_S2 = resnet50(pretrained=checkpoint, num_classes=1000)
        backbone_S2.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2),padding=(3, 3), bias=False)
        backbone_S2.fc = nn.Linear(2048, S2_num_features)
        
    elif network == "vit":
        vit_model = vit_b_16(pretrained=checkpoint)
        vit_model.heads.head = nn.Linear(768, S2_num_features)
        backbone_S2 = ViTWrapper(vit_model)
        
    elif network == "efficientnet_b0":
        backbone_S2 = efficientnet_b0(pretrained=checkpoint)
        backbone_S2.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        backbone_S2.classifier[1] = nn.Linear(1280, S2_num_features)
        
    # === 右腦：表格特徵萃取 (MLP Backbone) ===
    backbone_tabular = TabNetStyleBackbone(
        input_dim=tabular_input_count,   # 13 個特徵
        output_dim=tabular_features      # 輸出 64 維度
    )
    
    if ablation_mode == "both":
        shared_input_dim = S2_num_features + tabular_features # 640 + 64 = 704
    elif ablation_mode == "image_only":
        shared_input_dim = S2_num_features                    # 只吃影像的 640
    elif ablation_mode == "tabular_only":
        shared_input_dim = tabular_features                   # 只吃表格的 64
    else:
        raise ValueError("ablation_mode 必須是 'both', 'image_only' 或 'tabular_only'")
    
    # ... 前面的影像主幹選擇 ...

    # 🌟 完整的文本主幹頻譜切換
    if tabular_model == "linear":
        print("[INFO] 使用極致基礎：純線性表格主幹")
        backbone_tabular = Linear_TabularBackbone(tabular_input_count, tabular_features)
    elif tabular_model == "shallow_mlp":
        print("[INFO] 使用極致基礎：淺層 MLP 表格主幹")
        backbone_tabular = Shallow_MLP_TabularBackbone(tabular_input_count, tabular_features)
    elif tabular_model == "mlp":
        print("[INFO] 使用深度 MLP 表格主幹")
        backbone_tabular = MLP_TabularBackbone(tabular_input_count, tabular_features)
    elif tabular_model == "resnet":
        print("[INFO] 使用 ResNet 表格主幹")
        backbone_tabular = ResNet_TabularBackbone(tabular_input_count, tabular_features)
    else: # 預設 tabnet
        print("[INFO] 使用 TabNet-style 注意力表格主幹")
        backbone_tabular = TabNetStyleBackbone(tabular_input_count, tabular_features)

    # ... [後面的 Shared Head 與回傳邏輯保持不變] ...
    # ==========================================
    # 🌟 核心修改：Y型雙分支神經網路 (Multi-task Heads)
    # ==========================================
    
    # 1. 共同思考區 (Shared Layers)：將影像(640) + 表格(32) 融合並壓縮
    shared_head = nn.Sequential(
        nn.Linear(shared_input_dim, 384), # 🌟 根據模式動態決定輸入大小
        nn.ReLU(),
        nn.Linear(384, 128),
        nn.ReLU()
    )

    # 2. PM2.5 專屬專家 (PM2.5 Head)：接收 128 維特徵，只預測 PM2.5
    head_pm25 = nn.Sequential(
        nn.Linear(128, 64),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(16, 1) # 輸出 1 個數字
    )

    # 3. PM10 專屬專家 (PM10 Head)：接收 128 維特徵，只預測 PM10
    head_pm10 = nn.Sequential(
        nn.Linear(128, 64),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.Dropout(0.25),
        nn.ReLU(),
        nn.Linear(16, 1) # 輸出 1 個數字
    )

    # 組合模型
    regression_model = RegressionHead_3(
        backbone_S2, 
        backbone_tabular, 
        shared_head, 
        head_pm25, 
        head_pm10, 
        ablation_mode=ablation_mode  
    )

    return regression_model

class RegressionHead_3(nn.Module):
    # 🌟 接收 ablation_mode
    def __init__(self, backbone_S2, backbone_tabular, shared_head, head_pm25, head_pm10, ablation_mode="both"):
        super(RegressionHead_3, self).__init__()
        self.backbone_S2 = backbone_S2
        self.backbone_tabular = backbone_tabular
        self.shared_head = shared_head
        self.head_pm25 = head_pm25
        self.head_pm10 = head_pm10
        self.log_vars = nn.Parameter(torch.zeros(2))
        self.ablation_mode = ablation_mode # 🌟 儲存模式

    def forward(self, x):
        img = x.get("img")
        tabular = x.get("tabular")

        # 🌟 根據消融實驗模式決定資料流向
        if self.ablation_mode == "both":
            img_features = self.backbone_S2(img)           
            tab_features = self.backbone_tabular(tabular)  
            combined = torch.cat((img_features, tab_features), dim=1) 
        
        elif self.ablation_mode == "image_only":
            # 只提煉影像，捨棄表格資料
            combined = self.backbone_S2(img)
            
        elif self.ablation_mode == "tabular_only":
            # 只提煉表格，捨棄影像資料
            combined = self.backbone_tabular(tabular)

        # 後面的處理完全一樣
        shared_features = self.shared_head(combined)
        out_pm25 = self.head_pm25(shared_features).squeeze(dim=1)
        out_pm10 = self.head_pm10(shared_features).squeeze(dim=1)
        
        return out_pm25, out_pm10