import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPool2D, Flatten

train_path = 'F:/AgePredict/dataset/'
face_net = cv2.CascadeClassifier(
        'F:/AgePredict/haarcascade_frontalface_alt.xml')


def preprocessing(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='uint8')
    faces = face_net.detectMultiScale(gray)

    # 有找到臉
    if len(faces) >= 1:
        (x, y, w, h) = faces[0]
        face_img = img[y:y + h, x:x + w].copy()  # 把臉框出來
        face_img = cv2.resize(face_img, (64, 64), interpolation=cv2.INTER_LANCZOS4)
        face_img = np.array(face_img)
        face_img = face_img / 255.0
        return face_img
    else:
        return []


def make_dataset(X, y, path):
    progress = tqdm(total=len(os.listdir(path)))
    for idx, picture in enumerate(os.listdir(path)):
        progress.update(1)
        age = picture.split('_')[0]
        img = preprocessing(path + picture)

        # 有找到臉
        if len(img) >= 1:
            X.append(img)
            y.append(float(age))


X, y = [], []
make_dataset(X, y, train_path)

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

print("shape of training set:", X_train.shape)

model = Sequential()
input_shape=(X_train.shape[1], X_train.shape[1], 3)
model.add(Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='leaky_relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(4, 4)))
model.add(Dense(256, activation='leaky_relu'))

model.add(Conv2D(filters=2, kernel_size=(2, 2), padding='same', activation='leaky_relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dense(128, activation='leaky_relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='leaky_relu'))
model.add(Dense(64, activation='leaky_relu'))
model.add(Dense(1, activation='leaky_relu'))

callbacks = EarlyStopping( monitor="val_loss",patience=10)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=["accuracy"]
)

history = model.fit(X_train, y_train, epochs=200,callbacks = callbacks,
                    batch_size=8, validation_data=(X_val, y_val))

model.save('F:/AgePredict/age_prediction_model_rgb.h5')

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()