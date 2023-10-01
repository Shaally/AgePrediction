import os
from tqdm import tqdm, trange
import cv2
import numpy as np
from mlxtend.image import extract_face_landmarks

def composite_laplacian(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))
    img = cv2.bitwise_or(sobelx, sobely)
    img = 255 - img

    return img

def find_eye(img):
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

def find_mounth(img):
    top = landmarks[30]  # 找鼻尖
    down = landmarks[57]  # 找下唇
    left_len = int(min(2*(top[0] - landmarks[1][0]) / 3, 2*(top[0] - landmarks[2][0]) / 3, 2*(top[0] - landmarks[3][0]) / 3))  # 左臉頰長度
    right_len = int(min(2*(landmarks[13][0] - top[0]) / 3, 2*(landmarks[14][0] - top[0]) / 3, 2*(landmarks[15][0] - top[0]) / 3))  # 右臉頰長度
    left_up = [top[0] - left_len, top[1]]
    h, w = down[1] - top[1], left_len + right_len

    img[left_up[1]:left_up[1] + h, left_up[0]:left_up[0] + w] = comp_img[left_up[1]:left_up[1] + h, left_up[0]:left_up[0] + w]
    return img

train_path = 'F:/AgePredict/dataset/'

result_path = "F:/AgePredict/new_dataset/"
face_net = cv2.CascadeClassifier(
        'F:/AgePredict/haarcascade_frontalface_alt.xml')

progress = tqdm(total=len(os.listdir(train_path)))
for picture in os.listdir(train_path):
    progress.update(1)
    img = cv2.imread(train_path + picture)
    if picture in os.listdir(result_path):
        continue
    # img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.array(img, dtype='uint8')
    faces = face_net.detectMultiScale(gray)
    if len(faces) < 1:
        continue
    (x, y, w, h) = faces[0]
    img = img[y:y + h, x:x + w]
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    landmarks = extract_face_landmarks(img)
    if landmarks is not None:
        comp_img = composite_laplacian(img)
        img = find_eye(img)
        img = find_mounth(img)
        try:
            cv2.imwrite(result_path + picture, comp_img)
        except Exception:
            pass