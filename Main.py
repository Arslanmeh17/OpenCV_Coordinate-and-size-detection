from PyQt5.QtWidgets import *
from pyqt1deneme_python import Ui_MainWindow
from PyQt5.QtGui import QIntValidator, QDoubleValidator
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
import cv2

import cv2
import numpy as np
import os
import glob

import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np

import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np

import snap7
from snap7.util import *
from snap7.types import *
import time

def ReadMemory(plc, byte, bit, datatype):  # define read memory function
    result = plc.read_area(Areas['MK'], 0, byte, datatype)
    if datatype == S7WLBit:
        return get_bool(result, 0, 1)
    elif datatype == S7WLByte or datatype == S7WLWord:
        return get_int(result, 0)
    elif datatype == S7WLReal:
        return get_real(result, 0)
    elif datatype == S7WLDWord:
        return get_dword(result, 0)
    else:
        return None


def WriteMemory(plc, byte, bit, datatype, value):  # define write memory function
    result = plc.read_area(Areas['MK'], 0, byte, datatype)
    if datatype == S7WLBit:
        set_bool(result, 0, bit, value)
    elif datatype == S7WLByte or datatype == S7WLWord:
        set_int(result, 0, value)
    elif datatype == S7WLReal:
        set_real(result, 0, value)
    elif datatype == S7WLDWord:
        set_dword(result, 0, value)
    plc.write_area(Areas['MK'], 0, byte, result)


def WriteValues(xvall, yvall, thetavall, xadress, yadress, thetaadress):  # define write memory to specific adress function
    WriteMemory(plc, 2, 0, S7WLBit, True)  # write m2.0 True
    WriteMemory(plc, xadress, 0, S7WLReal, xvall)  # write x value to x adress
    WriteMemory(plc, yadress, 0, S7WLReal, yvall)  # write y value to y adress
    WriteMemory(plc, thetaadress, 0, S7WLReal, thetavall)  # write theta value to theta adress
    WriteMemory(plc, 2, 0, S7WLBit, False)  # write m2.0 False


IP = '192.168.0.1'  # IP plc
RACK = 0  # RACK PLC
SLOT = 1  # SLOT PLC

plc = snap7.client.Client()  # call snap7 client function
plc.connect(IP, RACK, SLOT)  # connect to plc

state = plc.get_cpu_state()  # read plc state run/stop/error
print(f'State:{state}')      # print state plc

Kx = 0.75
Ky = 0.98

global offsetx
global offsety
global offsetangle

offsetx = 0
offsety = 0
offsetangle = 0

class qt5denemeler(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.lineEdit.setText(str(Kx))
        self.ui.lineEdit_2.setText(str(Ky))

        self.ui.lineEdit.setEnabled(False)
        self.ui.lineEdit_2.setEnabled(False)
        self.ui.lineEdit_4.setEnabled(False)
        self.ui.lineEdit_3.setEnabled(False)
        self.ui.lineEdit_8.setEnabled(False)
        self.ui.lineEdit_9.setEnabled(False)
        self.ui.lineEdit_10.setEnabled(False)
        self.ui.lineEdit_11.setEnabled(False)
        self.ui.lineEdit_12.setEnabled(False)


        self.ui.lineEdit_5.setValidator(QIntValidator(-250,250,self))
        self.ui.lineEdit_6.setValidator(QIntValidator(-250,250,self))
        self.ui.lineEdit_7.setValidator(QIntValidator(-250,250,self))


        self.ui.lineEdit_5.returnPressed.connect(self.measure)
        self.ui.lineEdit_6.returnPressed.connect(self.measure)
        self.ui.lineEdit_7.returnPressed.connect(self.measure)

        self.ui.pushButton.clicked.connect(self.measure)

        self.ui.lineEdit_5.setText(str(offsetx))
        self.ui.lineEdit_6.setText(str(offsety))
        self.ui.lineEdit_7.setText(str(offsetangle))




    def start(self):

        cap = cv2.VideoCapture(1)

        while (cap.isOpened()):

            ret, frame = cap.read()

            if ret == True:
                self.displayImage(frame, 1)
                cv2.waitKey()

        cap.release()

        cv2.destroyAllWindows()

    def measure(self):

        offsetx = self.ui.lineEdit_5.text()
        offsety = self.ui.lineEdit_6.text()
        offsetangle = self.ui.lineEdit_7.text()

        offsetx = float(offsetx)
        offsety = float(offsety)
        offsetangle = float(offsetangle)

        # Defining the dimensions of checkerboard
        CHECKERBOARD = (6, 9)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        prev_img_shape = None

        # Extracting path of individual image stored in a given directory
        images = glob.glob('./foto/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display 
            them on the images of checker board
            """
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

            # cv2.imshow('img', img)
            # cv2.waitKey(0)

        # cv2.destroyAllWindows()

        h, w = img.shape[:2]

        """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
        """
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("Camera matrix : \n")
        print(mtx)
        print("dist : \n")
        print(dist)
        print("rvecs : \n")
        print(rvecs)
        print("tvecs : \n")
        print(tvecs)

        for fname in images:
            img = cv2.imread(fname)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            # Find the chess board corners
            print(h, w, dst.shape[:2])
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            dst = cv2.resize(dst, (0, 0), fx=0.5, fy=0.5)

        ##########################################################


        # Load the image
        cap = cv.VideoCapture(0)

        x = 0
        y = 0
        done = 0
        xbuffer = []
        ybuffer = []
        xsend = 0
        ysend = 0

        while (cap.isOpened()):

            ret, frame = cap.read()

            ret, frame = cap.read()
            img = frame
            img = cv2.undistort(img, mtx, dist, None, newcameramtx)

            readpermission = ReadMemory(plc, 10, 0, S7WLWord)  # read mw10.0

            # Was the image there?
            if img is None:
                print("Error: File not found")
                exit(0)

            # Convert image to grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Convert image to binary
            _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

            # Find all the contours in the thresholded image
            contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

            if readpermission == 1:

                for i, c in enumerate(contours):

                    # Calculate the area of each contour
                    area = cv.contourArea(c)

                    # Ignore contours that are too small or too large
                    if area < 15000 or 100000 < area:
                        continue

                    # cv.minAreaRect returns:
                    # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
                    rect = cv.minAreaRect(c)
                    box = cv.boxPoints(rect)
                    box = np.int0(box)

                    # Retrieve the key parameters of the rotated bounding box
                    center = (int(rect[0][0]), int(rect[0][1]))
                    width = int(rect[1][0])
                    height = int(rect[1][1])
                    angle = float(rect[2])
                    angle = round(angle, 3)

                    if width < height:
                        angle = 90 - angle
                    else:
                        angle = -angle

                    label = "  Rotation Angle: " + str(angle) + " degrees"
                    textbox = cv.rectangle(img, (center[0] - 35, center[1] - 25),
                                           (center[0] + 295, center[1] + 10), (255, 255, 255), -1)
                    cv.putText(img, label, (center[0] - 50, center[1]),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
                    cv.drawContours(img, [box], 0, (0, 0, 255), 2)
                    kx = 545 / 640
                    ky = 545 / 640
                    x = center[0] * kx
                    y = center[1] * ky

                    xbuffer.append(x)
                    ybuffer.append(y)

                    if len(xbuffer) == 20:
                        xbuffer.sort()
                        ybuffer.sort()

                        xbuffer.pop(0)
                        xbuffer.pop(0)
                        xbuffer.pop(0)
                        xbuffer.pop(0)
                        xbuffer.pop(0)
                        xbuffer.pop(0)
                        xbuffer.pop(0)
                        xbuffer.pop()
                        xbuffer.pop()
                        xbuffer.pop()
                        xbuffer.pop()
                        xbuffer.pop()
                        xbuffer.pop()
                        xbuffer.pop()

                        ybuffer.pop(0)
                        ybuffer.pop(0)
                        ybuffer.pop(0)
                        ybuffer.pop(0)
                        ybuffer.pop(0)
                        ybuffer.pop(0)
                        ybuffer.pop(0)
                        ybuffer.pop()
                        ybuffer.pop()
                        ybuffer.pop()
                        ybuffer.pop()
                        ybuffer.pop()
                        ybuffer.pop()
                        ybuffer.pop()

                        xsend = sum(xbuffer) / len(xbuffer)
                        ysend = sum(ybuffer) / len(ybuffer)

                        xsend = round(xsend, 3)
                        ysend = round(ysend, 3)

                        xsend = xsend + offsetx
                        ysend = ysend + offsety
                        angle = angle + offsetangle

                        self.ui.lineEdit_3.setText(str(xsend) + " mm")
                        self.ui.lineEdit_8.setText(str(ysend) + " mm")
                        self.ui.lineEdit_9.setText(str(angle) + " degree")
                        self.ui.lineEdit_10.setText(str(width) + " mm")
                        self.ui.lineEdit_11.setText(str(height) + " mm")
                        self.ui.lineEdit_12.setText(str(width*height) + " mm^2")

                        WriteValues(xsend, ysend, angle, 60, 70, 80)

                        xbuffer = []
                        ybuffer = []

            if ret == True:
                self.displayImage(img, 1)
                key = cv.waitKey(1)
                if key == 27:
                    break



        cap.release()
        cv2.destroyAllWindows()




    def displayImage(self, img, window=1):

        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.ui.label_5.setPixmap(QPixmap.fromImage(img))
        self.ui.label_5.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


    def process(self):
        offsetx = self.ui.lineEdit_5.text()
        offsety = self.ui.lineEdit_6.text()
        offsetangle = self.ui.lineEdit_7.text()

        offsetx = float(offsetx)
        offsety = float(offsety)
        offsetangle = float(offsetangle)




app = QApplication([])
window = qt5denemeler()
window.show()
app.exec_()






