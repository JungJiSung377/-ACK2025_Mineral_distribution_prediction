[README.md](https://github.com/user-attachments/files/21564783/README.md)
# **Prediction of Mineral Distribution Based on Drilling Core Color**
## **시추 코어 색상 기반 광물 분포 예측**


### 1. 요약

현재 전 세계적으로 국가 간 전략 광물 확보 경쟁이 치열해지며, 기존 시추 코어 분석 방식의 낮은 정밀도와 비효율성을 극복할 자동화된 AI 모델의 개발이 요구되었다. 따라서, 본 프로젝트는 시추 코어 이미지로부터 물성값과 광물 함량을 정량적으로 예측하는 2단계 딥러닝 회귀 모델을 개발하여 자원 탐사 효율을 획기적으로 높이고 국가적 자원 경쟁력 강화에 이바지 하고자 하였다.


### 2. 데이터셋 확보

![데이터셋 확보 이미지](https://github.com/JungJiSung377/-ACK2025_Mineral_distribution_prediction/blob/main/image.png)

  * 재료: 미국 지질조사국(U.S. Geological Survey, USGS) 2019-643-FA 프로젝트의 해저 퇴적물
  * 포함: 라인 스캔 코어 이미지(약 33.6 GB), MSCL(Multi-Sensor Core Logger) 물성 데이터


### 3. 데이터 전처리

1) ROI 추출 : 배경 제거 (색상 기반 마스킹 기법)
2) 분할 : 고정된 크기로 이미지 분할
3) 정답 매칭 : 깊이별 코어 이미지-MSCL 물성값 1:1 정렬
4) 정규화 : RGB 값 스케일 조정(학습 안정화)


## 시추 코어 내 물성값 예측


### 4. ResNet-18 Model

![데이터셋 확보 이미지](https://github.com/JungJiSung377/-ACK2025_Mineral_distribution_prediction/blob/main/image.png)

 * Residual Learning 구조와 Skip Connection을 적용해 깊은 신경망의 기울기 소실을 완화한 합성곱 신경망(CNN)
 * 사전 학습된 가중치를 기반으로 시추 코어 이미지(200×200픽셀)를 입력받아 3가지 물성값을 회귀 예측하도록 구성
 * ImageNet 사전 학습 가중치를 초깃값으로 활용하여 안정적인 학습을 유도


### 5. HRNet-DeepLabV3+ Model

