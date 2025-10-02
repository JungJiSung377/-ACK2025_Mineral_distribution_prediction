[README.md](https://github.com/user-attachments/files/21564783/README.md)
# **Prediction of Mineral Distribution Based on Drilling Core Color**
**시추 코어 색상 기반 광물 분포 예측**


### 1. 요약

현재 전 세계적으로 국가 간 전략 광물 확보 경쟁이 치열해지며, 기존 시추 코어 분석 방식의 낮은 정밀도와 비효율성을 극복할 자동화된 AI 모델의 개발이 요구되었다. 따라서, 본 프로젝트는 시추 코어 이미지로부터 물성값과 광물 함량을 정량적으로 예측하는 2단계 딥러닝 회귀 모델을 개발하여 자원 탐사 효율을 획기적으로 높이고 국가적 자원 경쟁력 강화에 이바지 하고자 하였다.


### 2. 데이터셋

![데이터셋 확보 image](https://github.com/JungJiSung377/-ACK2025_Mineral_distribution_prediction/blob/main/image2.png)

  * 재료: 미국 지질조사국(U.S. Geological Survey, USGS) 2019-643-FA 프로젝트의 해저 퇴적물
  * 포함: 라인 스캔 코어 이미지(약 33.6 GB), MSCL(Multi-Sensor Core Logger) 물성 데이터


### 3. 데이터 전처리

* 1) ROI 추출 : 배경 제거 (색상 기반 마스킹 기법)
* 2) 분할 : 고정된 크기로 이미지 분할
* 3) 정답 매칭 : 깊이별 코어 이미지-MSCL 물성값 1:1 정렬
* 4) 정규화 : RGB 값 스케일 조정(학습 안정화)


# 시추 코어 내 물성값 예측


### 4. ResNet-18 Model

![ResNet-18 Model image](https://github.com/JungJiSung377/-ACK2025_Mineral_distribution_prediction/blob/main/image.png)

 * Residual Learning 구조와 Skip Connection을 적용해 깊은 신경망의 기울기 소실을 완화한 합성곱 신경망(CNN)
 * 사전 학습된 가중치를 기반으로 시추 코어 이미지(200×200픽셀)를 입력받아 3가지 물성값을 회귀 예측하도록 구성
 * ImageNet 사전 학습 가중치를 초깃값으로 활용하여 안정적인 학습을 유도


### 5. HRNet-DeepLabV3+ Model

![HRNet-DeepLabV3+ Model image](https://github.com/JungJiSung377/-ACK2025_Mineral_distribution_prediction/blob/main/image3.png)

 * 여러 해상도의 이미지를 동시에 처리하며, 해상도 간의 반복적 정보 교환을 통해 고해상도 표현을 유지
 * HRNet을 백본(Backbone)으로 사용하여 다중 규모 특징을 추출한 후, DeepLabV3+의 ASPP(Atrous Spatial Pyramid Pooling) 모듈을 통해 다양한 수용 영역 정보를 통합한 뒤에 고수준, 저수준 피처를 결합해 정밀한 경계를 복원


### 6. 예측 및 회귀 성능평가 결과

![예측 결과 image](https://github.com/JungJiSung377/-ACK2025_Mineral_distribution_prediction/blob/main/image4.png)

 * 상단 세 개의 그래프: HRNet-DeepLabV3+ 모델 예측 결과 / 하단 세 개의 그래프: ResNet-18 모델 예측 결과
 * HRNet-DeepLabV3+ 모델이 고해상도와 저해상도 특징을 반복적으로 교환하고 다중 규모 정보를 활용하여 국소적 변화를 안정적으로 포착

![회귀 성능평가 결과 image](https://github.com/JungJiSung377/-ACK2025_Mineral_distribution_prediction/blob/main/image5.png)

 * MSE: 예측값과 실제값의 차이를 제곱한 값의 평균 / R^2은 모델 변동성 지표
 * HRNet-DeepLabV3+가 ResNet-18 모델에 비해 우수한 성능을 나타냄


# 시추 코어 내 광물 함량 예측

![시추 코어 내 광물 함량 예측 image](https://github.com/JungJiSung377/-ACK2025_Mineral_distribution_prediction/blob/main/image6.png)

 * 1) 영역 군집화(SLIC) : 유사 색상/공간 영역으로 1차 분할
 * 2) 정밀 분할(SAM) : 2차 정밀 분할/다수의 후보군 생성
 * 3) 함량 추정(NNLS 회귀) : 분할된 영역과 물성값 기반으로 광물 조성비(%) 회귀 추정


# 결론

 * 시추 코어 이미지 기반 광물의 물성값 예측 모델 : ResNet-18, HRNet-DeepLabV3+ 모델 개발 / 광물 함량 추정 모델 : NNLS 딥러닝 회귀 모델 개발
 * 암종 분류와 같은 정성적 분석에 그치던 국내 연구의 한계 극복
 * HRNet-DeepLabV3+ 모델의  예측 성능 우수성 : 시추 코어 이미지 분석에서의 고정밀 모델이 필수적임을 시사
 * NNLS 기법은 비선형적 광물 간 상호작용 충분히 반영 못하는 한계가 존재하기에 향후 비선형 혼합 모형이나 정규화 항을 추가하는 등 변형 기법으로의 확장 필요
 * 높은 효율성의 전략 광물 탐사 & 정밀한 3D 자원 모델링의 기반 마련으로 국가적 자원 경쟁력 강화에 이바지할 것으로 기대됨


# 향후 연구

 * 모델 고도화: 실제 자연은 더 복잡한 비선형적 상호작용이 발생, 예측 정확도와 물리적 현실성 발전시키고자 함
비선형 혼합 모델 도입: 정교한 물리 모델 결합하여, 광물 입자의 형태나 공극률 등 복잡한 변수까지 고려한 예측
 * 일반화 성능 확보: 특정 데이터셋을 넘어, 어떤 종류의 암석&탐사 환경에도 안정적 작동하도록 하고자 함
다양한 지질 데이터셋 확보 및 학습: 국내외 유관 기관과의 협력을 통해 다양한 암종 데이터 확보&학습


# 참고문헌

* International Energy Agency. The Role of C ritical Minerals in Clean Energy Transitions. Int
* U.S. Geological Survey. Critical Minerals. U.S. Geological Survey Report, 2023.
* Jentzen, A., Lüthi, R., and Maeder, M. Deep learning for lithological classification of drill cores using RGB images: A case study from Switzerland. Computers & Geosciences, Vol. 155, pp. 104876, 2021.
* Tate, M., Johnstone, R., and Kaur, J. Automated drill core scanning and machine learning for mineralogical analysis. Ore Geology Reviews, Vol. 143, pp.104719, 2022.
* Lindsay, J. M., and Van der Walt, M. C. G.“Automated lithological classification of drill core imagery using deep convolutional neural networks.” Computers & Geosciences, Vol. 155, pp.104880, 2021.
* 한국광해광업공단. 광물자원 보고서 2022. 한국광해광업공단, 2022.
* Yong, K., Choi, E., Han, G., and Oh, J. “Machine Learning for Non-destructive and Quantitative Mineral Analysis from Drill Core Images.” Minerals, Vol. 10, No. 12, pp.1090-1105, 2020.
* U.S. Geological Survey. Sediment core datafrom offshore southern Cascadia during field activity 2019-643-FA. U.S.Geological Survey Data Release, 2024.
* Boiger, R., Churakov, S. V., Ballester Llagaria, I., Kosakowski, G., Wüst, R., et al. Direct mineral content prediction from drillcore imagesvia transfer learning. Swiss Journal of Geosciences, Vol. 117, pp.1–15, 2024.
* Wang, J., et al. Deep High-Resolution Representation Learning for Visual Recognition. IEEE Transactions on PatternAnalysis and Machine Intelligence, Vol. 43, No. 10, pp.3351–3352, 2021.
* Chen, L. C., et al. Encoder-Decoder withAtrous Separable Convolution for Semantic Image Segmentation. EuropeanConference on Computer Vision (ECCV), LNCS, Vol. 11211, pp.833–842, 2018.
* Achanta, R., et al. SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. IEEE Transactions on PatternAnalysis and Machine Intelligence, Vol. 34, No. 11, pp.2275–2277, 2012.
* Kirillov, A., et al. Segment Anything. arXivpreprint, arXiv:2304.02643, 2023
