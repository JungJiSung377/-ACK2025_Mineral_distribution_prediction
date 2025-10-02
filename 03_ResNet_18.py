from google.colab import drive
drive.mount('/content/drive')

# 프로젝트 경로 설정
project_path = "/content/drive/MyDrive/core_project"
img_dir = f"{project_path}/core_images"
csv_dir = f"{project_path}/mscl_data"

!ls {project_path}
!ls {img_dir} | head
!ls {csv_dir} | head

!pip install torch torchvision pandas scikit-learn matplotlib Pillow tqdm

import pandas as pd
import numpy as np
import glob

# 모든 CSV 로드
csv_files = glob.glob(f"{csv_dir}/*.csv")
print(f"총 {len(csv_files)}개의 CSV 파일 로드")

dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    core_id = file.split("/")[-1].replace(".csv", "")
    df['core_id'] = core_id
    dataframes.append(df)

# 전체 합치기
full_df = pd.concat(dataframes, ignore_index=True)

# NaN 처리 (다항식 보간 + 전후값 보간)
columns_to_fix = ["pwave_vel", "density", "mag_sus"]
for col in columns_to_fix:
    if full_df[col].isna().sum() > 0:
        full_df[col] = full_df[col].interpolate(method="polynomial", order=2)
        full_df[col] = full_df[col].fillna(method="ffill").fillna(method="bfill")

print(full_df.head())

from PIL import Image
import os

# 매핑 키: core_id + section → 이미지 파일명
# 예시: 2019_643_FA_GC01_01.tif → core_id=2019_643_FA_GC01, section=01
# full_df에서 core_id와 depth 기반으로 가장 가까운 이미지 매칭

image_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
print(f"이미지 파일 개수: {len(image_files)}")

# 이미지 파일 → core_id 컬럼과 매핑
image_map = {}
for img in image_files:
    parts = img.replace(".tif", "").split("_")
    core_id = "_".join(parts[:4])  # 예: 2019_643_FA_GC01
    image_map.setdefault(core_id, []).append(img)

# 각 core_id의 이미지 리스트 확인
for k, v in list(image_map.items())[:3]:
    print(f"{k}: {v[:3]}")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

available_core_ids = set(image_map.keys())
full_df = full_df[full_df['core_id'].isin(available_core_ids)].reset_index(drop=True)

# ==========================================================================
# 수정 1 : csv 파일과 이미지 파일이 매칭되지 않은(한 쪽에 데이터가 없는 경우) 제외시킴
# ==========================================================================
# core_id를 통일된 형식으로 변경 (경로 제거)
full_df['core_id'] = full_df['core_id'].apply(lambda x: os.path.basename(x).replace(".csv", ""))

# 1. 공통된 core_id만 추출
csv_core_ids = set(full_df['core_id'].unique())
image_core_ids = set(image_map.keys())
common_core_ids = csv_core_ids & image_core_ids

# 2. 제외된 것들 확인
csv_only = csv_core_ids - image_core_ids
image_only = image_core_ids - csv_core_ids

print(f"\n❌ 제외된 core_id (CSV는 있지만 이미지 없음): {sorted(list(csv_only))}")
print(f"❌ 제외된 core_id (이미지는 있지만 CSV 없음): {sorted(list(image_only))}")

# 3. 공통된 core_id만 유지
filtered_df = full_df[full_df['core_id'].isin(common_core_ids)].reset_index(drop=True)
print(f"✅ 최종 usable 데이터 수: {len(filtered_df)}")

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================================================
# 수정 2 : 같은 코어 ID 이미지를 병합하고 1cm(200픽셀) 단위로 분할함.
# ============================================================
class CoreDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, pixels_per_cm=200):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.pixels_per_cm = pixels_per_cm
        self.samples = []

        for core_id in dataframe['core_id'].unique():
            df_core = dataframe[dataframe['core_id'] == core_id].reset_index(drop=True)
            img_list = image_map.get(core_id, [])
            if not img_list:
                continue

            '''
            이미지를 200 픽셀 단위(1cm = 200픽셀)로 잘라서 csv의 단위와 일치시킴
            이미지가 여러개로 나눠져 저장되어 있음. ()..._01.tif, ..._02.tif와 같이 저장되어 있음)
            따라서 같은 코어 ID를 가지고 있다면 이미지를 합쳐 200 픽셀 단위로 자름
            '''
            img_slices = []
            for img_name in sorted(img_list):  # 이미지 순서 보장
                img_path = os.path.join(img_dir, img_name)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        num_slices = height // self.pixels_per_cm
                        for i in range(num_slices):
                            img_slices.append((img_path, i))
                except:
                    continue

            usable = min(len(img_slices), len(df_core))
            for i in range(usable):
                self.samples.append({
                    "img_path": img_slices[i][0],
                    "slice_idx": img_slices[i][1],
                    "row": df_core.iloc[i]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        row = sample["row"]
        img_path = sample["img_path"]
        slice_idx = sample["slice_idx"]

        with Image.open(img_path) as img:
            x0, y0 = 0, slice_idx * self.pixels_per_cm
            x1, y1 = img.width, (slice_idx + 1) * self.pixels_per_cm
            crop = img.crop((x0, y0, x1, y1)).convert("RGB")

        if self.transform:
            crop = self.transform(crop)

        labels = torch.tensor([row["density"], row["pwave_vel"], row["mag_sus"]], dtype=torch.float32)

        return crop, labels

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from sklearn.model_selection import train_test_split

# ✅ 고유 core_id 기준으로 먼저 분할
core_ids = filtered_df["core_id"].unique()
train_ids, temp_ids = train_test_split(core_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.33, random_state=42)  # 0.3 * 0.33 ≈ 0.10

# ✅ 해당 ID들만 필터링
train_df = filtered_df[filtered_df["core_id"].isin(train_ids)].reset_index(drop=True)
val_df   = filtered_df[filtered_df["core_id"].isin(val_ids)].reset_index(drop=True)
test_df  = filtered_df[filtered_df["core_id"].isin(test_ids)].reset_index(drop=True)

# ✅ 다시 Dataset 및 DataLoader 생성
train_ds = CoreDataset(train_df, img_dir, transform)
val_ds   = CoreDataset(val_df, img_dir, transform)
test_ds  = CoreDataset(test_df, img_dir, transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

# ✅ 사전 학습된 ResNet18 + 회귀 헤드
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)  # 출력: density, pwave_vel, mag_sus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실함수 및 최적화 도구
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 스케일 정규화 (StandardScaler 기준) ➤ HRNet과 동일한 단위 정렬
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 학습용 라벨 스케일링 (fit)
all_labels = train_df[["density", "pwave_vel", "mag_sus"]].values
scaler.fit(all_labels)

# =====================
# 모델 학습 부분
# =====================
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3)  # density, pwave_vel, mag_sus

model = model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 15
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    print(f"\n🔁 Epoch [{epoch+1}/{num_epochs}]")
    train_bar = tqdm(train_loader, desc="🟢 Training", leave=False)

    for images, labels in train_bar:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch

# 모델 평가 모드
model.eval()
y_true, y_pred = [], []

# 🔄 예측값 수집
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        y_pred.append(outputs.cpu().numpy())   # 예측값 (정규화된 상태)
        y_true.append(labels.numpy())          # 실제값 (정규화된 상태)

# 배열 정리
y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

# 🔁 역정규화 (실제 단위로 환산)
y_true_denorm = scaler.inverse_transform(y_true)
y_pred_denorm = scaler.inverse_transform(y_pred)

# 📌 물성 이름
features = ['Density', 'P-wave Velocity', 'Magnetic Susceptibility']

# 📊 성능 지표 출력
print("\n📊 [실제 단위 기준으로 평가한 성능 지표]")
for i, name in enumerate(features):
    rmse = np.sqrt(mean_squared_error(y_true_denorm[:, i], y_pred_denorm[:, i]))
    mae = mean_absolute_error(y_true_denorm[:, i], y_pred_denorm[:, i])
    r2 = r2_score(y_true_denorm[:, i], y_pred_denorm[:, i])
    print(f"\n📌 {name}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

# 📈 꺾은선 그래프 (예측 vs 실제값 비교)
plt.figure(figsize=(18, 5))
for i, name in enumerate(features):
    plt.subplot(1, 3, i + 1)
    plt.plot(y_true_denorm[:, i], label="True", color='blue', marker='o', markersize=2, linewidth=1)
    plt.plot(y_pred_denorm[:, i], label="Pred", color='red', linestyle='--', marker='x', markersize=2, linewidth=1)
    plt.title(f"{name} (Slice 0 ~ {len(y_true) - 1})")
    plt.xlabel("Slice Index (1cm units)")
    plt.ylabel(name)
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

import pandas as pd
import os

# 예측 결과 CSV 저장
results_df = pd.DataFrame({
    "density": y_pred_denorm[:, 0],
    "pwave_vel": y_pred_denorm[:, 1],
    "mag_sus": y_pred_denorm[:, 2]
})
csv_save_path = "/content/drive/MyDrive/core_project/pred_resnet18_results.csv"
results_df.to_csv(csv_save_path, index=False)
print(f"✅ 예측 CSV 저장 완료: {csv_save_path}")

# 예측 vs 실제 시각화 (100 슬라이스 단위 저장)
def visualize_prediction_vs_truth_by_chunks(csv_path, output_dir, chunk_size=100, core_id='ResNet18_Core'):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    prop_names = ['density', 'pwave_vel', 'mag_sus']
    total_len = len(df)

    for start in range(0, total_len, chunk_size):
        end = min(start + chunk_size, total_len)
        x = list(range(start, end))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        for i, prop in enumerate(prop_names):
            y_true = df[f"true_{prop}"].iloc[start:end]
            y_pred = df[f"pred_{prop}"].iloc[start:end]

            axes[i].plot(x, y_true, 'o-', color='blue', label='True')
            axes[i].plot(x, y_pred, 'x--', color='red', label='Pred')
            axes[i].set_title(f"{prop.capitalize()} (Slice {start} ~ {end})")
            axes[i].set_xlabel("Slice Index (1cm units)")
            axes[i].set_ylabel(prop)
            axes[i].grid(True)
            axes[i].legend()

        save_path = os.path.join(output_dir, f"{core_id}_slice_{start}_{end}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 저장: {save_path}")
