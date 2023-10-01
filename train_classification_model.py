import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.layers import GaussianNoise
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, GlobalAveragePooling2D, Dense
import tensorflow as tf

train_path = 'D:/coding_practice/AgePrediction/dataset/train/'
test_path = 'D:/coding_practice/AgePrediction/dataset/test/'


def preprocessing(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    img = np.array(img)
    img = img / 255.0
    return img


age_dict = dict()
def make_dataset(path):
    X, y = [], []
    for age in os.listdir(path):
        for idx, img_path in enumerate(os.listdir(path + age)):
            if idx==3000:
                break
            img = preprocessing(path + age + '/' + img_path)
            X.append(img)
            if 0<=int(age)<=3:
                age_dict["0~3"] = 0
                y.append(0)
            elif 4<=int(age)<=7:
                age_dict["4~7"] = 1
                y.append(1)
            elif 8<=int(age)<=15:
                age_dict["8~15"] = 2
                y.append(2)
            elif 16<=int(age)<=24:
                age_dict["16~24"] = 3
                y.append(3)
            elif 25<=int(age)<=35:
                age_dict["25~35"] = 4
                y.append(4)
            elif 36<=int(age)<=50:
                age_dict["36~50"] = 5
                y.append(5)
            elif 51<=int(age)<=70:
                age_dict["51~70"] = 6
                y.append(6)
            else:
                age_dict["70~"] = 7
                y.append(7)

    X = np.array(X)
    y = np.array(y)
    return X, y

X, y = make_dataset(train_path)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
X_test, y_test = make_dataset(test_path)
print(X_train.shape)

model = ResNet50(input_shape=(X_train.shape[1], X_train.shape[1], 3), weights=None)

# model = Sequential()
# model.add(resnet)
# # model.add(GaussianNoise(0.1))
# model.add(Dropout(0.5))
# model.add(Dense(len(age_dict.keys()), activation='softmax'))

# sgd = optimizers.SGD(momentum=0.9, lr=0.003, decay=0.0001)
model.compile(loss="sparse_categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])

# model.compile(
#     optimizer='sgd',
#     loss='mse'
# )

history = model.fit(X_train, y_train, epochs=100,
                    batch_size=16, validation_data=(X_val, y_val))


model.save('F:/AgePredict/age_predict_model_all.h5')