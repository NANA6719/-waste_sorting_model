# 필요한 라이브러리 임포트
import os  # 운영체제와 상호작용을 위한 라이브러리
import numpy as np  # 수학적 연산을 위한 라이브러리
import matplotlib.pyplot as plt  # 그래프를 그리기 위한 라이브러리
from tensorflow.keras.applications import VGG16  # VGG16 모델을 사용하기 위한 라이브러리
from tensorflow.keras.layers import Dense, Flatten, Dropout  # 모델의 레이어 구성
from tensorflow.keras.models import Model  # Keras에서 모델을 정의하는데 사용
from tensorflow.keras.optimizers import Adam  # Adam 옵티마이저
import tensorflow as tf  # TensorFlow 라이브러리
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 이미지를 전처리하는 라이브러리

# 데이터 경로 설정 (훈련 데이터와 검증 데이터를 불러올 경로 설정)
validation_dataset_path = 'archive/garbegeImgSet_validation'  # 검증 데이터 폴더 경로
train_dataset_path = 'archive/garbegeImgSet_train'  # 훈련 데이터 폴더 경로

# 데이터 증강 설정 (훈련 데이터에 변형을 주어 모델의 일반화 성능을 향상시킴)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # 모든 픽셀 값을 0과 1 사이로 정규화
    rotation_range=40,  # 이미지를 무작위로 회전 (0~40도)
    width_shift_range=0.2,  # 이미지를 좌우로 20%까지 이동
    height_shift_range=0.2,  # 이미지를 상하로 20%까지 이동
    shear_range=0.2,  # 이미지를 무작위로 전단 변형 (왜곡)
    zoom_range=0.2,  # 이미지를 무작위로 확대/축소
    horizontal_flip=True  # 이미지를 무작위로 좌우 반전
)

# 검증 데이터는 증강 없이 단순히 정규화만 함
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# 훈련 데이터 생성기 (디렉토리에서 이미지를 로드하여 모델 훈련에 사용할 데이터 생성)
train_generator = train_datagen.flow_from_directory(
    train_dataset_path,  # 훈련 데이터가 저장된 폴더 경로
    target_size=(224, 224),  # 모든 이미지를 224x224 크기로 리사이즈
    batch_size=32,  # 한번에 불러올 이미지 배치 크기
    class_mode='categorical',  # 다중 클래스 분류 문제
    shuffle=True  # 훈련 데이터를 섞어서 학습에 도움이 되도록
)

# 검증 데이터 생성기 (디렉토리에서 이미지를 로드하여 검증 데이터 생성)
validation_generator = validation_datagen.flow_from_directory(
    validation_dataset_path,  # 검증 데이터가 저장된 폴더 경로
    target_size=(224, 224),  # 모든 이미지를 224x224 크기로 리사이즈
    batch_size=32,  # 한번에 불러올 이미지 배치 크기
    class_mode='categorical'  # 다중 클래스 분류 문제
)

# 전이 학습을 위한 VGG16 모델 로드
# VGG16 모델은 ImageNet 데이터셋에서 학습된 모델로, include_top=False로 설정하여 
# 최상위 분류층을 제외하고, 특성 추출을 위한 모델만 가져옵니다.
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# VGG16의 출력층 위에 새로 학습할 층 추가
x = base_model.output
x = Flatten()(x)  # 출력을 1D 배열로 평탄화
x = Dense(512, activation='relu')(x)  # 512개의 뉴런을 가진 fully connected layer 추가
x = Dropout(0.5)(x)  # 50% 비율로 뉴런을 랜덤하게 제외하여 과적합을 방지
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)  # 클래스 수에 맞게 출력층 추가 (softmax 활성화 함수)

# 새로운 모델 정의 (기존 VGG16 모델에 우리가 추가한 층을 덧붙임)
model = Model(inputs=base_model.input, outputs=predictions)

# VGG16의 층을 동결하여 전이 학습
# VGG16의 기존 층을 학습하지 않고, 우리가 추가한 층만 학습하도록 설정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=1e-4),  # Adam 옵티마이저, 학습률을 1e-4로 설정
              loss='categorical_crossentropy',  # 다중 클래스 분류를 위한 손실 함수
              metrics=['accuracy'])  # 정확도를 평가 지표로 사용

# 모델 훈련
history = model.fit(
    train_generator,  # 훈련 데이터
    epochs=20,  # 훈련을 20번 반복
    validation_data=validation_generator  # 검증 데이터
)

# 모델 평가
# 훈련이 끝난 후 검증 데이터를 이용해 성능을 평가
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation loss: {loss:.2f}')  # 검증 손실 출력
print(f'Validation accuracy: {accuracy:.2f}')  # 검증 정확도 출력

# 훈련 과정 시각화
# 정확도와 손실 그래프를 출력하여 학습이 잘 진행되었는지 시각적으로 확인
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')  # 훈련 정확도 그래프
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # 검증 정확도 그래프
plt.title('Model Accuracy')  # 그래프 제목
plt.xlabel('Epoch')  # x축 레이블
plt.ylabel('Accuracy')  # y축 레이블
plt.legend()  # 범례 추가

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')  # 훈련 손실 그래프
plt.plot(history.history['val_loss'], label='Validation Loss')  # 검증 손실 그래프
plt.title('Model Loss')  # 그래프 제목
plt.xlabel('Epoch')  # x축 레이블
plt.ylabel('Loss')  # y축 레이블
plt.legend()  # 범례 추가

plt.show()  # 그래프 출력

# 모델 저장 경로 설정
model_save_path = 'recycling_model.h5'

# 훈련이 끝난 후 모델을 파일로 저장
model.save(model_save_path)  # 모델 저장
print(f"Model saved to {model_save_path}")  # 저장된 경로 출력
