import cv2
import sys
import time
import math
import numpy as np
from keras.models import load_model
from mlxtend.image import extract_face_landmarks
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from UI.menu_ui import Menu_Ui_Form
from UI.image_ui import Image_Ui_Form
from UI.video_ui import Video_Ui_Form


class MainWindows(QDialog):

    def __init__(self):
        super().__init__(None)
        self.Main = Menu_Ui_Form()
        self.Main.setupUi(self)


# 選擇輸入圖片(Image)可能會做的行為
class Image:
    def __init__(self):
        self.img = None
        self.faces = None

    def load_sample(self, image_path):
        self.img = cv2.imread(image_path)

        h, w = self.img.shape[1], self.img.shape[0]

        while h < 600 and w < 600:
            h *= 2
            w *= 2

        while h > 600 and w > 600:
            h /= 2
            w /= 2

        h, w = int(h), int(w)
        self.img = cv2.resize(self.img, (h, w), interpolation=cv2.INTER_LANCZOS4)

    # 用 haarcascade_frontalface_alt.xml 找照片中的臉
    def find_face(self, face_net):
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # haarcascade_frontalface_alt 的輸入需要灰階圖像
        detect = np.array(self.gray, dtype='uint8')
        self.faces = face_net.detectMultiScale(detect)

        print("find ", len(self.faces), " people")

    # 產生訓練時的圖片
    def composite_laplacian(self, img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        sobelx = np.uint8(np.absolute(sobelx))
        sobely = np.uint8(np.absolute(sobely))
        img = cv2.bitwise_or(sobelx, sobely)
        img = 255 - img

        return img

    def add_composite_laplacian(self, img, comp_img):
        # find eye
        left, right = self.landmarks[36], self.landmarks[45]
        up = min(self.landmarks[37][1], self.landmarks[38][1], self.landmarks[43][1], self.landmarks[44][1])
        down = max(self.landmarks[40][1], self.landmarks[41][1], self.landmarks[46][1], self.landmarks[47][1])

        left[0] -= 7
        right[0] += 7
        up -= 5
        down += 7

        h, w = down - up, right[0] - left[0]

        left_up = [left[0], up]

        img[left_up[1]:left_up[1] + h, left_up[0]:left_up[0] + w] = comp_img[left_up[1]:left_up[1] + h,
                                                                    left_up[0]:left_up[0] + w]

        # find mouth
        top = self.landmarks[30]  # 找鼻尖
        down = self.landmarks[57]  # 找下唇
        left_len = int(min(2 * (top[0] - self.landmarks[1][0]) / 3, 2 * (top[0] - self.landmarks[2][0]) / 3,
                           2 * (top[0] - self.landmarks[3][0]) / 3))  # 左臉頰長度
        right_len = int(min(2 * (self.landmarks[13][0] - top[0]) / 3, 2 * (self.landmarks[14][0] - top[0]) / 3,
                            2 * (self.landmarks[15][0] - top[0]) / 3))  # 右臉頰長度
        left_up = [top[0] - left_len, top[1]]
        h, w = down[1] - top[1], left_len + right_len

        img[left_up[1]:left_up[1] + h, left_up[0]:left_up[0] + w] = comp_img[left_up[1]:left_up[1] + h,
                                                                    left_up[0]:left_up[0] + w]
        return img

    # 預測圖片中所有臉的年齡
    def predict_each_age(self, net):
        # faces 紀錄所有臉的位置
        for (x, y, w, h) in self.faces:

            cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 255, 255), 2)  # 把臉框出來
            face_img = self.img[y:y + h, x:x + w].copy()
            face_img = cv2.resize(face_img, (64, 64), interpolation=cv2.INTER_LANCZOS4)
            # self.landmarks = extract_face_landmarks(self.img)
            # if self.landmarks is None:
            #     continue
            # input_image = self.composite_laplacian(face_img)
            # input_image = self.add_composite_laplacian(face_img, input_image)
            input_image = face_img.reshape(1, 64, 64, 3)  # 調整成訓練模型的 input維度
            input_image = np.array(input_image)/255.0
            age_preds = net.predict(input_image)
            print(age_preds)

            age_num = math.floor(age_preds[0][0])
            # 設定輸出年齡
            if age_num <= 2:
                age = "0~2"
            elif 3 <= age_num <= 12:
                age = "3~12"
            elif 13 <= age_num <= 19:
                age = "13~19"
            elif 20 <= age_num <= 29:
                age = "20~29"
            elif 30 <= age_num <= 39:
                age = "30~39"
            elif 40 <= age_num <= 49:
                age = "40~49"
            elif 50 <= age_num <= 59:
                age = "50~59"
            elif 60 <= age_num <= 69:
                age = "60~69"
            elif 70 <= age_num <= 79:
                age = "70~79"
            elif 80 <= age_num <= 89:
                age = "80~89"
            elif age_num >= 90:
                age = "older than 90"
            else:
                age = ""

            overlay_text = "%s" % (age)
            cv2.putText(self.img, overlay_text, (x, y), 2, 1, (0, 150, 255), 2, cv2.LINE_AA)

        return self.img


# 選擇輸入圖片(Image)會看到的介面設定
class ImageUI(QDialog):

    def __init__(self, net, face_net):
        super().__init__(None)
        self.Image_UI = Image_Ui_Form()
        self.Image_UI.setupUi(self)

        # 當按下 Choose Image 按鈕執行 get_image function 取得圖片
        self.Image_UI.pushButton.clicked.connect(self.get_image)

        self.net = net
        self.face_net = face_net

        self.label = QLabel("")  # 放圖片的地方
        self.grid = QGridLayout()
        self.grid.addWidget(self.Image_UI.pushButton)  # Choose Image button
        self.grid.addWidget(self.label)
        self.grid.addWidget(self.Image_UI.pushButton_2)  # Back to Menu button

    # 開啟檔案選擇圖片後用模型預測年齡，並把預測後的結果畫在圖片上
    def get_image(self):
        # 開檔案選照片
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'D:/coding_practice/AgePrediction/picture/',
                                            "Image files (*.jpg *.png)")
        if fname[0] == '':  # 被直接關掉
            pass
        else:
            image_path = fname[0]
            img_obj = Image()  # 建立 Image 物件執行預測年齡行為
            img_obj.load_sample(image_path)  # 載入圖片並 resize
            img_obj.find_face(self.face_net)  # 找出所有的臉
            img = img_obj.predict_each_age(self.net)  # 預測每一個臉

            w, h, channel = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換成 RGB彩色圖像
            qimg = QImage(img.data, h, w, 3 * h, QImage.Format_RGB888)  # 把用 numpy表示的圖片轉成 QImage格式
            pixmap = QPixmap(qimg)  # 轉換成 QtGui 物件

            self.label.setPixmap(QPixmap(pixmap))  # 把圖片加入介面
            self.resize(pixmap.width(), pixmap.height())
            self.setLayout(self.grid)  # 設定輸出
            self.show()


# 選擇輸入影像(Video)可能會做的行為
# 繼承 Image(部分功能可以不用重新寫)，差別在於 Video要在 while迴圈中顯示
class Video(QtCore.QThread, Image):
    rawdata = QtCore.pyqtSignal(np.ndarray)  # 建立 pyqtSignal 物件，用來傳遞影像訊息

    def __init__(self,net, face_net, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, 600)  # set width of the frame
        self.cap.set(4, 800)  # set height of the frame
        self.net = net
        self.face_net = face_net

        # WebCam 沒有打開
        if self.cap is None or not self.cap.isOpened():
            self.connect = False
            self.running = False
        # WebCam 已經打開
        else:
            self.connect = True
            self.running = False

    # 讀影像資訊後用模型預測年齡，並把預測後的結果畫在影像上
    def run(self):
        while self.running and self.connect:
            self.ret, img = self.cap.read()

            # 可以直接用 Image 的 find_face、predict_each_age function
            self.img = img
            self.find_face(self.face_net)
            self.predict_each_age(self.net)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

            # 有讀到WebCam的影像
            if self.ret:
                self.rawdata.emit(self.img)    # 發送影像
            else:
                print("Warning!!!")
                self.connect = False

    # 開啟 WebCam 影像讀取功能
    def open(self):
        if self.connect:
            self.running = True  # 啟動讀取狀態

    # 暫停影像讀取功能
    def stop(self):
        if self.connect:
            self.running = False  # 關閉讀取狀態

    # 關閉影像讀取功能
    def close(self):
        if self.connect:
            self.running = False
            time.sleep(1)
            self.cap.release()


# 選擇輸入影像(WebCam)會看到的介面設定
class VideoUI(QtWidgets.QMainWindow, Video_Ui_Form):

    def __init__(self, net, face_net, parent=None):
        super(VideoUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.viewData.setScaledContents(True)  # 設定圖片自動適應視窗大小
        self.net = net
        self.face_net = face_net
        self.ProcessCam = Video(self.net, self.face_net)  # 建立一個相機物件的 thread

        # WebCam 已經打開
        if self.ProcessCam.connect:
            self.ProcessCam.rawdata.connect(self.showData)  # 取得並顯示影像
            self.pushButton.setEnabled(True)  # start button可以按
        # WebCam 還沒打開
        else:
            self.pushButton.setEnabled(False)

        self.pushButton_2.setEnabled(False)  # stop button不能按

        self.pushButton.clicked.connect(self.openCam)  # start:開啟攝影機
        self.pushButton_2.clicked.connect(self.stopCam)  # stop:暫停讀取影像
        self.pushButton_3.clicked.connect(self.stopCam)  # Back to Menu:暫停讀取影像

    # 打開相機
    def openCam(self):
        if self.ProcessCam.connect:  # 判斷攝影機是否可用
            self.ProcessCam.open()   # 影像讀取功能開啟
            self.ProcessCam.start()  # 在子緒啟動影像讀取
            self.pushButton.setEnabled(False)
            self.pushButton_2.setEnabled(True)

    # 關閉相機
    def stopCam(self):
        if self.ProcessCam.connect:
            self.ProcessCam.stop()
            self.pushButton.setEnabled(True)
            self.pushButton_2.setEnabled(False)

    # 把圖片放到介面上
    def showData(self, img):
        self.Ny, self.Nx, _ = img.shape

        qimg = QtGui.QImage(img.data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)

        self.viewData.setScaledContents(True)
        self.viewData.setPixmap(QtGui.QPixmap.fromImage(qimg))

    # 視窗應用程式關閉事件
    def closeEvent(self, event):
        if self.ProcessCam.running:
            self.ProcessCam.close()      # 關閉攝影機
            time.sleep(1)
            self.ProcessCam.terminate()  # 關閉子緒
        QtWidgets.QApplication.closeAllWindows()  # 關閉所有視窗

    # 鍵盤事件
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:   # 偵測是否按下鍵盤 Q
            if self.ProcessCam.running:
                self.ProcessCam.close()      # 關閉攝影機
                time.sleep(1)
                self.ProcessCam.terminate()  # 關閉子緒
            QtWidgets.QApplication.closeAllWindows()  # 關閉所有視窗


if __name__ == '__main__':
    # age predict model
    net = load_model('D:/coding_practice/AgePrediction/model/age_classification_model_weight.h5')
    # face detect model
    face_net = cv2.CascadeClassifier(
        'D:/coding_practice/AgePrediction/Age-Gender_Prediction-master/haarcascade_frontalface_alt.xml')

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindows()
    main.show()

    image_ui = ImageUI(net, face_net)
    video_ui = VideoUI(net, face_net)

    main.Main.pushButton.clicked.connect(
        lambda: {main.close(), image_ui.show()}
    )
    main.Main.pushButton_2.clicked.connect(
        lambda: {main.close(), video_ui.show()}
    )
    image_ui.Image_UI.pushButton_2.clicked.connect(
        lambda: {image_ui.close(), main.show()}
    )
    video_ui.pushButton_3.clicked.connect(
        lambda: {video_ui.close(), main.show()}
    )
    sys.exit(app.exec_())