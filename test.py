import os
import cv2
import numpy as np
from keras.models import load_model
from mlxtend.image import extract_face_landmarks


model_rgb = load_model('D:/coding_practice/AgePrediction/model/age_prediction_model_rgb.h5')
model_gray = load_model('D:/coding_practice/AgePrediction/model/age_prediction_model_gray_face_feature.h5')
face_net = cv2.CascadeClassifier(
        'D:/coding_practice/AgePrediction/Age-Gender_Prediction-master/haarcascade_frontalface_alt.xml')
test_path = 'D:/coding_practice/AgePrediction/dataset/test/'


def preprocessing_rgb(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='uint8')
    faces = face_net.detectMultiScale(gray)

    # 有找到臉
    if len(faces) >= 1:
        (x, y, w, h) = faces[0]
        face_img = img[y:y + h, x:x + w].copy()  # 把臉框出來
        face_img = cv2.resize(face_img, (64, 64), interpolation=cv2.INTER_LANCZOS4)
        face_img = np.array(face_img)
        face_img = face_img.reshape(1, 64, 64, 3)
        face_img = face_img / 255.0
        return face_img
    else:
        return []


def preprocessing_gray(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.array(gray, dtype='uint8')
    faces = face_net.detectMultiScale(gray)
    if len(faces)>=1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = faces[0]
        img = img[y:y + h, x:x + w]
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LANCZOS4)
        landmarks = extract_face_landmarks(img)
        if landmarks is not None:
            comp_img = composite_laplacian(img)
            img = find_eye(img, comp_img, landmarks)
            img = find_mounth(img, comp_img, landmarks)
            # img = np.array(img, dtype='float64')
            # cv2.imshow('My Image', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img.reshape(1, 64, 64, 3)
            img = img / 255.0
            return img
        else:
            return []
    else:
        return []

def composite_laplacian(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))
    img = cv2.bitwise_or(sobelx, sobely)
    img = 255 - img

    return img


def find_eye(img, comp_img, landmarks):
    left, right = landmarks[36], landmarks[45]
    up = min(landmarks[37][1], landmarks[38][1], landmarks[43][1], landmarks[44][1])
    down = max(landmarks[40][1], landmarks[41][1], landmarks[46][1], landmarks[47][1])

    left[0] -= 7
    right[0] += 7
    up -= 5
    down += 7

    h, w = down-up, right[0]-left[0]

    left_up = [left[0], up]

    img[left_up[1]:left_up[1] + h, left_up[0]:left_up[0] + w] = comp_img[left_up[1]:left_up[1]+h, left_up[0]:left_up[0]+w]
    return img


def find_mounth(img, comp_img, landmarks):
    top = landmarks[30]  # 找鼻尖
    down = landmarks[57]  # 找下唇
    left_len = int(min(2*(top[0] - landmarks[1][0]) / 3, 2*(top[0] - landmarks[2][0]) / 3, 2*(top[0] - landmarks[3][0]) / 3))  # 左臉頰長度
    right_len = int(min(2*(landmarks[13][0] - top[0]) / 3, 2*(landmarks[14][0] - top[0]) / 3, 2*(landmarks[15][0] - top[0]) / 3))  # 右臉頰長度
    left_up = [top[0] - left_len, top[1]]
    h, w = down[1] - top[1], left_len + right_len

    img[left_up[1]:left_up[1] + h, left_up[0]:left_up[0] + w] = comp_img[left_up[1]:left_up[1] + h, left_up[0]:left_up[0] + w]
    return img

mae_rgb, num_rgb, mae_gray, num_gray = 0, 0, 0, 0
for age in os.listdir(test_path):
    print(age)
    for sample in os.listdir(test_path + age):
        input_img_rgb = preprocessing_rgb(test_path + age + '/' + sample)
        input_img_gray = preprocessing_gray(test_path + age + '/' + sample)
        if len(input_img_rgb) >= 1:
            age_preds = model_rgb.predict(input_img_rgb)
            mae_rgb += abs(age_preds-int(age))
            num_rgb += 1
        if len(input_img_gray) >= 1:
            age_preds = model_gray.predict(input_img_gray)
            mae_gray += abs(age_preds-int(age))
            num_gray += 1
    print("RGB MAE:", mae_rgb / num_rgb)
    print("RGB MAE:", mae_gray / num_gray)

print("RGB MAE:", mae_rgb/num_rgb)
print("RGB MAE:", mae_gray/num_gray)