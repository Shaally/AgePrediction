import cv2
import sys
import time
import numpy as np
from keras.models import load_model
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from UI.menu_ui import Menu_Ui_Form
from UI.image_ui import Image_Ui_Form
from UI.video_ui import Video_Ui_Form

class MainWindows(QDialog):
    def __init__(self):
        super().__init__()
        self.Main = Menu_Ui_Form()
        self.Main.setupUi(self)


class ImageUI(QDialog):
    def __init__(self, net, face_net):
        super().__init__()
        self.Image_UI = Image_Ui_Form()
        self.Image_UI.setupUi(self)
        self.Image_UI.pushButton.clicked.connect(self.get_image)
        self.net = net
        self.face_net = face_net

        self.grid = QGridLayout()
        self.grid.addWidget(self.Image_UI.pushButton)

        self.label = QLabel("Hello")
        self.grid.addWidget(self.label)
        self.grid.addWidget(self.Image_UI.pushButton_2)

    def get_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'D:/coding_practice/AgePrediction/',
                                            "Image files (*.jpg *.png)")
        if fname[0] != '':
            image_path = fname[0]
            img_obj = Image()
            img = img_obj.load_sample(image_path)
            img_obj.find_face(img, self.face_net)
            img = img_obj.predict_each_age(img, self.net)

            w, h, ch = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            qimg = QImage(img.data, h, w, 3 * h, QImage.Format_RGB888)
            pixmap = QPixmap(qimg)

            self.label.setPixmap(QPixmap(pixmap))
            self.resize(pixmap.width(), pixmap.height())

            self.setLayout(self.grid)

            self.show()


class Image:

    # def __init__(self, image_path):
    #     self.image_path = image_path

    def load_sample(self, image_path):
        img = cv2.imread(image_path)
        h, w = img.shape[1], img.shape[0]
        while h > 600 and w > 600:
            h /= 2
            w /= 2
        h = int(h)
        w = int(w)
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_LANCZOS4)

        return img

    def find_face(self, img, face_net):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray, dtype='uint8')
        self.faces = face_net.detectMultiScale(gray)

        print("find ", len(self.faces), " people")

    def predict_age(self, img, net, x, y, w, h):
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # Get Face
        face_img = img[y:y + h, h:h + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 3, (64, 64), swapRB=False)
        blob = blob.reshape(1, 64, 64, 3)
        print(net.predict([blob]))
        age_preds = np.mean(net.predict([blob]))
        print(age_preds)
        age = "wrong"
        if 0 <= age_preds <= 5:
            age = "0~5"
        elif 6 <= age_preds <= 10:
            age = "6~10"
        elif 11 <= age_preds <= 15:
            age = "11~15"
        elif 16 <= age_preds <= 20:
            age = "16~20"
        elif 21 <= age_preds <= 25:
            age = "21~25"
        elif 26 <= age_preds <= 30:
            age = "26~30"
        elif 31 <= age_preds <= 35:
            age = "31~35"
        elif 36 <= age_preds <= 40:
            age = "36~40"
        elif 41 <= age_preds <= 50:
            age = "41~50"
        elif 51 <= age_preds <= 55:
            age = "51~55"

        print(age)
        overlay_text = "%s" % (age)
        cv2.putText(img, overlay_text, (x, y), 1, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return img

    def predict_each_age(self, img, net):
        for (x, y, w, h) in self.faces:
            img = self.predict_age(img, net, x, y, w, h)

        return img


class VideoUI(QtWidgets.QMainWindow, Video_Ui_Form):
    def __init__(self, net, face_net, parent=None):

        super(VideoUI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.viewData.setScaledContents(True)
        self.net = net
        self.face_net = face_net

        self.ProcessCam = Camera(self.net, self.face_net)  # 建立相機物件

        if self.ProcessCam.connect:
            self.ProcessCam.rawdata.connect(self.getRaw)  # 槽功能：取得並顯示影像
            self.pushButton.setEnabled(True)
        else:
            self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)

        self.pushButton.clicked.connect(self.openCam)  # 槽功能：開啟攝影機
        self.pushButton_2.clicked.connect(self.stopCam)  # 槽功能：暫停讀取影像
        self.pushButton_3.clicked.connect(self.stopCam)  # 槽功能：暫停讀取影像

    def getRaw(self, data):  # data 為接收到的影像
        self.showData(data)  # 將影像傳入至 showData()

    def openCam(self):
        if self.ProcessCam.connect:  # 判斷攝影機是否可用
            self.ProcessCam.open()   # 影像讀取功能開啟
            self.ProcessCam.start()  # 在子緒啟動影像讀取
            self.pushButton.setEnabled(False)
            self.pushButton_2.setEnabled(True)

    def stopCam(self):
        if self.ProcessCam.connect:
            self.ProcessCam.stop()
            self.pushButton.setEnabled(True)
            self.pushButton_2.setEnabled(False)

    def showData(self, img):
        self.Ny, self.Nx, _ = img.shape
        print(img.shape)

        qimg = QtGui.QImage(img.data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)

        self.viewData.setScaledContents(True)
        self.viewData.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        """ 視窗應用程式關閉事件 """
        if self.ProcessCam.running:
            self.ProcessCam.close()      # 關閉攝影機
            time.sleep(1)
            self.ProcessCam.terminate()  # 關閉子緒
        QtWidgets.QApplication.closeAllWindows()  # 關閉所有視窗

    def keyPressEvent(self, event):
        """ 鍵盤事件 """
        if event.key() == QtCore.Qt.Key_Q:   # 偵測是否按下鍵盤 Q
            if self.ProcessCam.running:
                self.ProcessCam.close()      # 關閉攝影機
                time.sleep(1)
                self.ProcessCam.terminate()  # 關閉子緒
            QtWidgets.QApplication.closeAllWindows()  # 關閉所有視窗


class Camera(QtCore.QThread, Image):
    rawdata = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, net, face_net, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, 600)  # set width of the frame
        self.cap.set(4, 800)  # set height of the frame
        self.net = net
        self.face_net = face_net

        # self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.cap is None or not self.cap.isOpened():
            self.connect = False
            self.running = False
        else:
            self.connect = True
            self.running = False

    def find_face(self):
        self.ret, self.img = self.cap.read()
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray, dtype='uint8')
        self.faces = self.face_net.detectMultiScale(gray)

        print("find ", len(self.faces), " people")

    def predict_each_age(self):
        print(self.faces)
        for (x, y, w, h) in self.faces:
            self.img = self.predict_age(self.img, self.net, x, y, w, h)
            # cv2.imshow("frame", self.img)

    def run(self):
        while self.running and self.connect:
            self.find_face()
            self.predict_each_age()
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            if self.ret:
                self.rawdata.emit(self.img)    # 發送影像
            else:
                print("Warning!!!")
                self.connect = False

    def open(self):
        """ 開啟攝影機影像讀取功能 """
        if self.connect:
            self.running = True    # 啟動讀取狀態

    def stop(self):
        if self.connect:
            self.running = False

    def close(self):
        if self.connect:
            self.running = False
            time.sleep(1)
            self.cap.release()


if __name__ == '__main__':
    net = load_model('D:/coding_practice/AgePrediction/model/age_prediction_model_rgb.h5')
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
    # main.Main.pushButton_2.clicked.connect(
    #     lambda: {main.show(), Video(net, face_net).start()}
    # )
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