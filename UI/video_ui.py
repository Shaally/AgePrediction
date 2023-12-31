# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'video.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Video_Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("WebCam")
        Form.resize(800, 600)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(50, 80, 381, 241))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayoutWidget.resize(700, 480)

        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.viewData = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.viewData.setText("")
        self.viewData.setObjectName("viewData")
        self.verticalLayout.addWidget(self.viewData)

        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(30, 10, 56, 17))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.resize(100, 50)

        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(150, 10, 56, 17))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.resize(100, 50)

        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(630, 10, 71, 20))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.resize(150, 50)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("WebCam", "WebCam"))
        self.pushButton.setText(_translate("WebCam", "Open"))
        self.pushButton_2.setText(_translate("WebCam", "Stop"))
        self.pushButton_3.setText(_translate("WebCam", "Back to Menu"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Video_Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
