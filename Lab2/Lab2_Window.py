# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Lab2_Window.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
import numbers

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QDoubleValidator
from PyQt5.QtWidgets import QFileDialog
from PIL import Image
from scipy.signal import butter, lfilter


class Ui_Lab2_Window(object):
    imageAdded = False
    src = np.zeros((200, 200, 4), np.uint8)
    isJpg = False

    def display_Image(self, fileName):
        global spectrum_image
        self.src = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
        # afficher l'image originale
        pixmap = QPixmap(fileName)
        self.label_3.setPixmap(pixmap)
        self.label_7.setPixmap(pixmap)  # Frquency Original Image
        self.label_31.setPixmap(pixmap)  # Canny Original Image
        self.imageAdded = True

        # afficher l'image spectrale originale
        if (self.imageAdded):
            self.spectrum_image = self.get_spectrum_image(self.src)
            # Affichage de l'image transformée
            cv2.imwrite('spectrum.jpg', self.spectrum_image)
            pixmap = QPixmap('spectrum.jpg')
            self.label_13.setPixmap(pixmap)  # Original Spectrum Image
            # self.label_8.setPixmap(pixmap) # Ideal Low-Pass reconstructed Image 1

    def openImage(self):
        global gaussian_filter
        global sobel_horiz_filter
        global sobel_vert_filter
        global sobel_filter
        global laplace_4_filter
        global laplace_8_filter
        global custom_filter
        # read image from file dialog window
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.centralwidget, "Open Image", "",
                                                  "Images (*.jpg);;Images (*.png);;All Files (*)", options=options)
        try:
            i = Image.open(fileName)
            if (i.format == 'JPEG'):
                self.isJpg = True
                self.display_Image(fileName)
            if (i.format == 'PNG'):
                self.isJpg = False
                self.display_Image(fileName)
            if (i.format != 'PNG' and (i.format != 'JPEG')):
                print('no valid type')
            self.gaussian_filter = self.gaussian_kernel(3, sigma=np.sqrt(3), verbose=True)
            self.sobel_filter = self.gaussian_kernel(9, sigma=np.sqrt(9), verbose=True)
            self.sobel_horiz_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            self.sobel_vert_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            self.laplace_4_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            self.laplace_8_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

        except IOError:
            pass

    def filterChanged(self):
        type_filter = str(self.comboBox_5.currentText())
        mean_val = "{}".format(0.11)
        coeff_1 = ''
        coeff_2 = ''
        coeff_3 = ''
        coeff_4 = ''
        coeff_5 = ''
        coeff_6 = ''
        coeff_7 = ''
        coeff_8 = ''
        coeff_9 = ''

        if (type_filter != 'Costum'):
            if (type_filter == 'Mean'):
                coeff_1 = mean_val;
                coeff_2 = mean_val;
                coeff_3 = mean_val
                coeff_4 = mean_val;
                coeff_5 = mean_val;
                coeff_6 = mean_val
                coeff_7 = mean_val;
                coeff_8 = mean_val;
                coeff_9 = mean_val

            elif type_filter == 'Gaussian':
                coeff_1 = "{}".format(self.gaussian_filter[0][0])
                coeff_2 = "{}".format(self.gaussian_filter[0][1])
                coeff_3 = "{}".format(self.gaussian_filter[0][2])
                coeff_4 = "{}".format(self.gaussian_filter[1][0])
                coeff_5 = "{}".format(self.gaussian_filter[1][1])
                coeff_6 = "{}".format(self.gaussian_filter[1][2])
                coeff_7 = "{}".format(self.gaussian_filter[2][0])
                coeff_8 = "{}".format(self.gaussian_filter[2][1])
                coeff_9 = "{}".format(self.gaussian_filter[2][2])
            elif type_filter == '4 - Neighbour Laplacian':
                coeff_1 = "{}".format(self.laplace_4_filter[0][0])
                coeff_2 = "{}".format(self.laplace_4_filter[0][1])
                coeff_3 = "{}".format(self.laplace_4_filter[0][2])
                coeff_4 = "{}".format(self.laplace_4_filter[1][0])
                coeff_5 = "{}".format(self.laplace_4_filter[1][1])
                coeff_6 = "{}".format(self.laplace_4_filter[1][2])
                coeff_7 = "{}".format(self.laplace_4_filter[2][0])
                coeff_8 = "{}".format(self.laplace_4_filter[2][1])
                coeff_9 = "{}".format(self.laplace_4_filter[2][2])

            elif type_filter == '8 - Neighbour Laplacian':
                coeff_1 = "{}".format(self.laplace_8_filter[0][0])
                coeff_2 = "{}".format(self.laplace_8_filter[0][1])
                coeff_3 = "{}".format(self.laplace_8_filter[0][2])
                coeff_4 = "{}".format(self.laplace_8_filter[1][0])
                coeff_5 = "{}".format(self.laplace_8_filter[1][1])
                coeff_6 = "{}".format(self.laplace_8_filter[1][2])
                coeff_7 = "{}".format(self.laplace_8_filter[2][0])
                coeff_8 = "{}".format(self.laplace_8_filter[2][1])
                coeff_9 = "{}".format(self.laplace_8_filter[2][2])

            elif type_filter == 'Sobel Horiz':
                coeff_1 = "{}".format(self.sobel_horiz_filter[0][0])
                coeff_2 = "{}".format(self.sobel_horiz_filter[0][1])
                coeff_3 = "{}".format(self.sobel_horiz_filter[0][2])
                coeff_4 = "{}".format(self.sobel_horiz_filter[1][0])
                coeff_5 = "{}".format(self.sobel_horiz_filter[1][1])
                coeff_6 = "{}".format(self.sobel_horiz_filter[1][2])
                coeff_7 = "{}".format(self.sobel_horiz_filter[2][0])
                coeff_8 = "{}".format(self.sobel_horiz_filter[2][1])
                coeff_9 = "{}".format(self.sobel_horiz_filter[2][2])

            elif type_filter == 'Sobel Vert':
                coeff_1 = "{}".format(self.sobel_vert_filter[0][0])
                coeff_2 = "{}".format(self.sobel_vert_filter[0][1])
                coeff_3 = "{}".format(self.sobel_vert_filter[0][2])
                coeff_4 = "{}".format(self.sobel_vert_filter[1][0])
                coeff_5 = "{}".format(self.sobel_vert_filter[1][1])
                coeff_6 = "{}".format(self.sobel_vert_filter[1][2])
                coeff_7 = "{}".format(self.sobel_vert_filter[2][0])
                coeff_8 = "{}".format(self.sobel_vert_filter[2][1])
                coeff_9 = "{}".format(self.sobel_vert_filter[2][2])

            elif type_filter == 'Sobel':
                coeff_1 = "{}".format(self.sobel_filter[0][0])
                coeff_2 = "{}".format(self.sobel_filter[0][1])
                coeff_3 = "{}".format(self.sobel_filter[0][2])
                coeff_4 = "{}".format(self.sobel_filter[1][0])
                coeff_5 = "{}".format(self.sobel_filter[1][1])
                coeff_6 = "{}".format(self.sobel_filter[1][2])
                coeff_7 = "{}".format(self.sobel_filter[2][0])
                coeff_8 = "{}".format(self.sobel_filter[2][1])
                coeff_9 = "{}".format(self.sobel_filter[2][2])

            # Update
            self.lineEdit.setText(coeff_1)
            self.lineEdit_2.setText(coeff_2)
            self.lineEdit_3.setText(coeff_3)
            self.lineEdit_4.setText(coeff_4)
            self.lineEdit_5.setText(coeff_5)
            self.lineEdit_6.setText(coeff_6)
            self.lineEdit_7.setText(coeff_7)
            self.lineEdit_8.setText(coeff_8)
            self.lineEdit_9.setText(coeff_9)
            self.lineEdit.setEnabled(False)
            self.lineEdit_2.setEnabled(False)
            self.lineEdit_3.setEnabled(False)
            self.lineEdit_4.setEnabled(False)
            self.lineEdit_5.setEnabled(False)
            self.lineEdit_6.setEnabled(False)
            self.lineEdit_7.setEnabled(False)
            self.lineEdit_8.setEnabled(False)
            self.lineEdit_9.setEnabled(False)
        else:
            self.lineEdit.setText('')
            self.lineEdit_2.setText('')
            self.lineEdit_3.setText('')
            self.lineEdit_4.setText('')
            self.lineEdit_5.setText('')
            self.lineEdit_6.setText('')
            self.lineEdit_7.setText('')
            self.lineEdit_8.setText('')
            self.lineEdit_9.setText('')
            self.lineEdit.setEnabled(True)
            self.lineEdit_2.setEnabled(True)
            self.lineEdit_3.setEnabled(True)
            self.lineEdit_4.setEnabled(True)
            self.lineEdit_5.setEnabled(True)
            self.lineEdit_6.setEnabled(True)
            self.lineEdit_7.setEnabled(True)
            self.lineEdit_8.setEnabled(True)
            self.lineEdit_9.setEnabled(True)

    def right(self, str, size):
        return str[-size:]

    def applyIdealLowPassFilter(self):

        # Récupération des paramètres  -  Exemple notation d0=50
        value = str(self.lineEdit_10.text())
        param = float(self.right(value, len(value)-3))

        if (isinstance(param, numbers.Number) != True):
            param = 50.

        img = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        d0 = param if isinstance(param, numbers.Number) else 100
        [M, N] = img.shape
        H = img.copy()

        # On centre l'image
        # for i in range(M):
        #     for j in range(N):
        #         #H[i][j] = pow((-1), (H[i][0]+H[0][j]))
        #         H[i][j] = pow((-1), (i + j))

        # Calcul de la transformée de Fourier
        fftI = np.fft.fft2(H)
        fftI = np.fft.fftshift(fftI).real
        # fftI = 20 * np.log(np.abs(fftI))
        #print(fftI)

        # Construction du filtre
        for u in range(M):
            for v in range(N):
                dist = np.sqrt(np.square(u - M / 2) + np.square(v - N / 2))
                #print(dist)
                H[u][v] = 1. if dist <= d0 else 0.

        # Mutltipication de l'image transformée avec le filtre
        image_spectrale = np.real(fftI) * H

        # Affichage de l'image transformée
        cv2.imwrite('spectrum2.jpg', image_spectrale)
        pixmap = QPixmap('spectrum2.jpg')
        self.label_14.setPixmap(pixmap)  # Ideal Low-Pass reconstructed Image 1
        tmp_image = image_spectrale.copy()

        # Calcul de la transformée inverse de Fourier de l'image spectrale filtrée
        ifftI = np.fft.ifft2(tmp_image).real
        #image_filtree = np.real(np.fft.ifftshift(ifftI))
        #print(ifftI)

        # Affichage de l'image filtrée
        cv2.imwrite('ilpfilter.jpg', ifftI)
        pixmap = QPixmap('ilpfilter.jpg')
        self.label_8.setPixmap(pixmap)  # Ideal Low-Pass reconstructed Image 1


    def butter_lowpass_spectrum(self, p1, p2):

        # Conversion de l'image en grayscale
        img = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        H = img.copy()
        H = cv2.GaussianBlur(H, (3, 3), 0)
        d0 = p1
        n = p2
        [M, N] = H.shape

        # Calcul de la transforméée de Fourier de l'image
        f = np.fft.fft2(H)
        fshift = np.fft.fftshift(f)
        fshift = 20 * np.log(np.abs(fshift))

        # Calcul du filtre butterworth
        for u in range(M):
            for v in range(N):
                    dist = np.sqrt(np.square(u - M / 2) + np.square(v - N / 2))
                    # Création du filtre Butterworth
                    H[u][v] = 1 / (1 + pow((dist / d0), (2 * n)))

        # print(H)
        # Mutltipication de l'image transformée avec le filtre
        image_spectrale = np.real(fshift) * H

        # Affichage de l'image transformée
        cv2.imwrite('Butterworth_spec.jpg', image_spectrale)
        pixmap = QPixmap('Butterworth_spec.jpg')
        self.label_15.setPixmap(pixmap)  # Low-Pass Butterworth reconstructed Image 1


    def butter_lowpass_filter(self, data, cutoff, fs, degre=5):
        b, a = self.butter_lowpass(cutoff, fs, degre=degre)
        y = lfilter(b, a, data)
        return y

    def butter_lowpass(self, cutoff, fs, degre=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(degre, normal_cutoff, btype='low', analog=False)
        return b, a

    def applyButterworthFilter(self):
        print("Very Good !")
        img = self.src  # imread('grass.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Récupération des paramètres - Exemple notation d0=50;n=100
        value = str(self.lineEdit_11.text()).split(";")
        param_1 = float(self.right(value[0], len(value[0]) - 3))
        param_2 = float(self.right(value[1], len(value[1]) - 2))

        if (isinstance(param_1, numbers.Number) != True):
            param_1 = 50.
            param_2 = 30.

        # Filter requirements.
        degre = 6
        fs = param_1  # taux d'échantillonnage, Hz
        cutoff = 3.667  # fréquence de coupure souhaitée du filtre (en Hz)

        img_filtered = self.butter_lowpass_filter(img, cutoff, fs, degre)

        # Affichage de l'image transformée
        cv2.imwrite('Butterworth.jpg', img_filtered)
        pixmap = QPixmap('Butterworth.jpg')
        self.label_9.setPixmap(pixmap)  # Low-Pass Butterworth reconstructed Image 1

        self.butter_lowpass_spectrum(param_1, param_2)

    def applyFilter(self):
        global isOk
        isOk = False
        if (str(self.comboBox_5.currentText()) == 'Costum') \
                and (str(self.lineEdit.text()) == '' or str(self.lineEdit_2.text()) == '' or
                     str(self.lineEdit_3.text()) == '' or str(self.lineEdit_4.text()) == '' or
                     str(self.lineEdit_5.text()) == '' or str(self.lineEdit_6.text()) == '' or
                     str(self.lineEdit_7.text()) == '' or str(self.lineEdit_8.text()) == '' or
                     str(self.lineEdit_9.text()) == ''):
            print("Veuillez ne laisser aucune case filtre !")
            pass
        else:
            self.custom_filter = np.zeros((3, 3))
            self.custom_filter[0][0] = float(str(self.lineEdit.text()))
            self.custom_filter[0][1] = float(str(self.lineEdit_2.text()))
            self.custom_filter[0][2] = float(str(self.lineEdit_3.text()))
            self.custom_filter[1][0] = float(str(self.lineEdit_4.text()))
            self.custom_filter[1][1] = float(str(self.lineEdit_5.text()))
            self.custom_filter[1][2] = float(str(self.lineEdit_6.text()))
            self.custom_filter[2][0] = float(str(self.lineEdit_7.text()))
            self.custom_filter[2][1] = float(str(self.lineEdit_8.text()))
            self.custom_filter[2][2] = float(str(self.lineEdit_9.text()))
            self.custom_filter = np.array(self.custom_filter)
            isOk = True
            print(self.custom_filter)

        if (str(self.comboBox_5.currentText()) == 'Mean' and str(self.comboBox_7.currentText()) == '0' and str(
                self.comboBox_6.currentText()) == 'Clamp 0 ... 255' and self.imageAdded):
            # appliquer le filtre moyenneur 3*3
            blur = cv2.blur(self.src, (3, 3))
            # clamp 0 ... 255
            self.clamb_0_to_255(blur, "Mean")

            # border
            self.setBorder(blur, "blurred_Gauss", "Mean")

        # (Copy or 0) border strategy and (Normalize 0 to 255 or Clamp 0 ... 255) range strategy
        if ((str(self.comboBox_7.currentText()) == 'Copy' or str(self.comboBox_7.currentText()) == '0')
                and (str(self.comboBox_6.currentText()) == 'Normalize 0 to 255' or str(
                    self.comboBox_6.currentText()) == 'Clamp 0 ... 255')
                and self.imageAdded):

            if str(self.comboBox_5.currentText()) == 'Gaussian':
                print("Gaussian")
                my_kernel = np.array([[0.0625, 0.125, 0.0625],
                                      [0.125, 0.25, 0.125],
                                      [0.0625, 0.125, 0.0625]])

                blur = self.gaussian_blur(self.src, my_kernel, 3, verbose=True, isCustom=False)
                # cv2.GaussianBlur(self.src, (3, 3), 0)  #

                # Clamp 0 ... 255
                if str(self.comboBox_6.currentText()) == 'Clamp 0 ... 255':  # Clamp 0 ... 255
                    print("Clamp 0 ... 255 Strategy")
                    blur = self.clamb_0_to_255(blur, "Gaussian")

                    cv2.imwrite('blurred_Gauss.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Gauss.jpg')
                    self.label_21.setPixmap(pixmap)

                # Normalize 0 to 255
                elif str(self.comboBox_6.currentText()) == 'Normalize 0 to 255':  # Normalize 0 to 255
                    print("Normalize 0 to 255 Strategy")
                    blur = self.normalize_to_255(blur)

                    cv2.imwrite('blurred_Gauss.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Gauss.jpg')
                    self.label_21.setPixmap(pixmap)

                # BORDER
                if str(self.comboBox_7.currentText()) == 'Copy':
                    print("Mirror Strategy")
                    blur = self.copy(blur)

                    cv2.imwrite('blurred_Gauss.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Gauss.jpg')
                    self.label_21.setPixmap(pixmap)
                elif str(self.comboBox_7.currentText()) == '0':
                    print("0 Strategy")
                    self.setBorder(blur, "blurred_Gauss", 'Gaussian')

            elif str(self.comboBox_5.currentText()) == '4 - Neighbour Laplacian':
                print("4 - Neighbour Laplacian")
                # converting to gray scale
                gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

                # remove noise
                tmp_image = cv2.GaussianBlur(gray, (3, 3), 0)

                abs_dst = cv2.Laplacian(tmp_image, cv2.CV_8UC2)

                # converting back to uint8
                blur = cv2.convertScaleAbs(abs_dst)

                if str(self.comboBox_6.currentText()) == 'Clamp 0 ... 255':  # Clamp 0 ... 255
                    print("Clamp 0 ... 255 Strategy")
                    blur = self.clamb_0_to_255(blur, "laplace_4")

                    cv2.imwrite('blurred_laplace_4.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_laplace_4.jpg')
                    self.label_21.setPixmap(pixmap)

                # Normalize 0 to 255
                elif str(self.comboBox_6.currentText()) == 'Normalize 0 to 255':  # Normalize 0 to 255
                    print("Normalize 0 to 255 Strategy")
                    blur = self.normalize_to_255(blur)

                    cv2.imwrite('blurred_laplace_4.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_laplace_4.jpg')
                    self.label_21.setPixmap(pixmap)

                # BORDER
                if str(self.comboBox_7.currentText()) == 'Copy':
                    print("Mirror Strategy")
                    blur = self.copy(blur)

                    cv2.imwrite('blurred_laplace.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_laplace_4.jpg')
                    self.label_21.setPixmap(pixmap)
                elif str(self.comboBox_7.currentText()) == '0':
                    print("0 Strategy")
                    self.setBorder(blur, "blurred_laplace", "laplace_4")

            elif str(self.comboBox_5.currentText()) == '8 - Neighbour Laplacian':
                print("8 - Neighbour Laplacian")
                # converting to gray scale
                gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

                # remove noise
                tmp_image = cv2.GaussianBlur(gray, (3, 3), 0)

                abs_dst = cv2.Laplacian(tmp_image, cv2.CV_64F)

                # converting back to uint8
                blur = cv2.convertScaleAbs(abs_dst)

                if str(self.comboBox_6.currentText()) == 'Clamp 0 ... 255':  # Clamp 0 ... 255
                    print("Clamp 0 ... 255 Strategy")
                    blur = self.clamb_0_to_255(blur, "laplace_8")

                    cv2.imwrite('blurred_laplace_8.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_laplace_8.jpg')
                    self.label_21.setPixmap(pixmap)

                # Normalize 0 to 255
                elif str(self.comboBox_6.currentText()) == 'Normalize 0 to 255':  # Normalize 0 to 255
                    print("Normalize 0 to 255 Strategy")
                    blur = self.normalize_to_255(blur)

                    cv2.imwrite('blurred_laplace_8.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_laplace_8.jpg')
                    self.label_21.setPixmap(pixmap)

                # BORDER
                if str(self.comboBox_7.currentText()) == 'Copy':
                    print("Mirror Strategy")
                    blur = self.copy(blur)

                    cv2.imwrite('blurred_laplace_8.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_laplace_8.jpg')
                    self.label_21.setPixmap(pixmap)
                elif str(self.comboBox_7.currentText()) == '0':
                    print("0 Strategy")
                    self.setBorder(blur, "blurred_laplace", "laplace_8")

            elif str(self.comboBox_5.currentText()) == 'Sobel Horiz':
                print("Sobel Horiz")
                # converting to gray scale
                gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

                # remove noise
                tmp_image = cv2.GaussianBlur(gray, (3, 3), 0)

                # my_kernel = np.zeros((0, 0, 3), np.uint8)  # No importance !
                # tmp_image = self.gaussian_blur(self.src, my_kernel, 3, verbose=True, isCustom=False)
                # blur = self.sobel_edge_detection(tmp_image, self.sobel_horiz_filter, "sobel_horiz", verbose=True)
                blur = cv2.Sobel(tmp_image, -1, 1, 0)
                # Clamp 0 ... 255
                if str(self.comboBox_6.currentText()) == 'Clamp 0 ... 255':  # Clamp 0 ... 255
                    print("Clamp 0 ... 255 Strategy")
                    blur = self.clamb_0_to_255(blur, "Sobel_horiz")

                    cv2.imwrite('blurred_Sobel_horiz.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Sobel_horiz.jpg')
                    self.label_21.setPixmap(pixmap)

                # Normalize 0 to 255
                elif str(self.comboBox_6.currentText()) == 'Normalize 0 to 255':  # Normalize 0 to 255
                    print("Normalize 0 to 255 Strategy")
                    blur = self.normalize_to_255(blur)

                    cv2.imwrite('blurred_Sobel_horiz.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Sobel_horiz.jpg')
                    self.label_21.setPixmap(pixmap)

                # BORDER
                if str(self.comboBox_7.currentText()) == 'Copy':
                    print("Mirror Strategy")
                    blur = self.copy(blur)

                    cv2.imwrite('blurred_Sobel_horiz.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Sobel_horiz.jpg')
                    self.label_21.setPixmap(pixmap)
                elif str(self.comboBox_7.currentText()) == '0':
                    print("0 Strategy")
                    self.setBorder(blur, "blurred_Sobel", 'Sobel')
            elif str(self.comboBox_5.currentText()) == 'Sobel Vert':
                print("Sobel Vert")
                # converting to gray scale
                gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

                # remove noise
                tmp_image = cv2.GaussianBlur(gray, (3, 3), 0)

                # my_kernel = np.zeros((0, 0, 3), np.uint8)  # No importance !
                # tmp_image = self.gaussian_blur(self.src, my_kernel, 3, verbose=True, isCustom=False)
                # blur = self.sobel_edge_detection(tmp_image, self.sobel_horiz_filter, "sobel_vert", verbose=True)
                blur = cv2.Sobel(tmp_image, -1, 0, 1)

                # Clamp 0 ... 255
                if str(self.comboBox_6.currentText()) == 'Clamp 0 ... 255':  # Clamp 0 ... 255
                    print("Clamp 0 ... 255 Strategy")
                    blur = self.clamb_0_to_255(blur, "Sobel_vert")

                    cv2.imwrite('blurred_Sobel_vert.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Sobel_vert.jpg')
                    self.label_21.setPixmap(pixmap)

                # Normalize 0 to 255
                elif str(self.comboBox_6.currentText()) == 'Normalize 0 to 255':  # Normalize 0 to 255
                    print("Normalize 0 to 255 Strategy")
                    blur = self.normalize_to_255(blur)

                    cv2.imwrite('blurred_Sobel_vert.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Sobel_vert.jpg')
                    self.label_21.setPixmap(pixmap)

                # BORDER
                if str(self.comboBox_7.currentText()) == 'Copy':
                    print("Mirror Strategy")
                    blur = self.copy(blur)

                    cv2.imwrite('blurred_Sobel_vert.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Sobel_vert.jpg')
                    self.label_21.setPixmap(pixmap)
                elif str(self.comboBox_7.currentText()) == '0':
                    print("0 Strategy")
                    self.setBorder(blur, "blurred_Sobel", 'Sobel')
            elif str(self.comboBox_5.currentText()) == 'Sobel':
                print("Sobel")
                # converting to gray scale
                gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

                # remove noise
                tmp_image = cv2.GaussianBlur(gray, (3, 3), 0)
                # my_kernel = np.zeros((0, 0, 3), np.uint8)   # No importance !
                # tmp_image = self.gaussian_blur(self.src, my_kernel, 3, verbose=True, isCustom=False)

                blur = self.sobel_detection_contours(tmp_image, self.sobel_horiz_filter, "sobel", verbose=True)

                # Clamp 0 ... 255
                if str(self.comboBox_6.currentText()) == 'Clamp 0 ... 255':  # Clamp 0 ... 255
                    print("Clamp 0 ... 255 Strategy")
                    blur = self.clamb_0_to_255(blur, "Sobel")

                    cv2.imwrite('blurred_Sobel.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Sobel.jpg')
                    self.label_21.setPixmap(pixmap)

                # Normalize 0 to 255
                elif str(self.comboBox_6.currentText()) == 'Normalize 0 to 255':  # Normalize 0 to 255
                    print("Normalize 0 to 255 Strategy")
                    blur = self.normalize_to_255(blur)

                    cv2.imwrite('blurred_Sobel.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Sobel.jpg')
                    self.label_21.setPixmap(pixmap)

                # BORDER
                if str(self.comboBox_7.currentText()) == 'Copy':
                    print("Mirror Strategy")
                    blur = self.copy(blur)

                    cv2.imwrite('blurred_Sobel.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Sobel.jpg')
                    self.label_21.setPixmap(pixmap)
                elif str(self.comboBox_7.currentText()) == '0':
                    print("0 Strategy")
                    self.setBorder(blur, "blurred_Sobel", 'Sobel')

            elif str(self.comboBox_5.currentText()) == 'Costum':
                # converting to gray scale
                gray = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)

                # remove noise
                my_kernel = np.zeros((0, 0, 3), np.uint8)  # No importance !
                tmp_image = self.gaussian_blur(self.src, my_kernel, 3, verbose=True, isCustom=False)

                blur = self.convolution(tmp_image, self.custom_filter, average=True, verbose=False)

                # my_kernel = np.zeros((0, 0, 3), np.uint8)  # No importance !
                # blur = self.sobel_edge_detection(tmp_image, self.custom_filter, "custom", verbose=True)

                # Clamp 0 ... 255
                if str(self.comboBox_6.currentText()) == 'Clamp 0 ... 255':  # Clamp 0 ... 255
                    print("Clamp 0 ... 255 Strategy")
                    blur = self.clamb_0_to_255(blur, "Custom")

                    cv2.imwrite('blurred_Custom.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Custom.jpg')
                    self.label_21.setPixmap(pixmap)

                # Normalize 0 to 255
                elif str(self.comboBox_6.currentText()) == 'Normalize 0 to 255':  # Normalize 0 to 255
                    print("Normalize 0 to 255 Strategy")
                    blur = self.normalize_to_255(blur)

                    cv2.imwrite('blurred_Custom.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Custom.jpg')
                    self.label_21.setPixmap(pixmap)

                # BORDER
                if str(self.comboBox_7.currentText()) == 'Copy':
                    print("Mirror Strategy")
                    blur = self.copy(blur)

                    cv2.imwrite('blurred_Custom.jpg', blur)
                    # afficher l'image filtrée
                    pixmap = QPixmap('blurred_Custom.jpg')
                    self.label_21.setPixmap(pixmap)
                elif str(self.comboBox_7.currentText()) == '0':
                    print("0 Strategy")
                    self.setBorder(blur, "blurred_Custom", 'Costum')

    def get_spectrum_image(self, image):

        # Conversion de l'image en grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calcul de la transforméée de Fourier de l'image
        f = np.fft.fft2(gray_image)
        fshift = np.fft.fftshift(f)
        image_spectrum = 20 * np.log(np.abs(fshift))

        return image_spectrum

    def copy(self, filter_image):
        return cv2.copyMakeBorder(filter_image, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    def normalize_to_255(self, filter_image):
        return cv2.normalize(filter_image, filter_image, 0, 255, cv2.NORM_MINMAX)

    def setBorder(self, threated_image, img_name, filter_type):
        if (filter_type == "Mean"):
            # jpg image has 3 channels
            if (self.isJpg == True):
                threated_image[0] = [0, 0, 0]  # ligne y = 0
                threated_image[threated_image.shape[0] - 1] = [0, 0, 0]  # ligne y = (height - 1)
                threated_image[:, 0] = [0, 0, 0]  # colonne x = 0
                threated_image[:, threated_image.shape[1] - 1] = [0, 0, 0]  # colonne y = (width - 1)
                cv2.imwrite(img_name + ".jpg", threated_image)
                # afficher l'image filtrée
                pixmap = QPixmap(img_name + ".jpg")
                self.label_21.setPixmap(pixmap)
            # png image has 4 channels
            if (self.isJpg == False):
                threated_image[0] = [0, 0, 0, 255]  # ligne y = 0
                threated_image[threated_image.shape[0] - 1] = [0, 0, 0, 255]  # ligne y = (height - 1)
                threated_image[:, 0] = [0, 0, 0, 255]  # colonne x = 0
                threated_image[:, threated_image.shape[1] - 1] = [0, 0, 0, 255]  # colonne y = (width - 1)
                cv2.imwrite(img_name + ".png", threated_image)
                # afficher l'image filtrée
                pixmap = QPixmap(img_name + ".png")
                self.label_21.setPixmap(pixmap)
        else:
            # jpg image has 3 channels
            if (self.isJpg == True):
                threated_image[0] = [0]  # ligne y = 0
                threated_image[threated_image.shape[0] - 1] = [0]  # ligne y = (height - 1)
                threated_image[:, 0] = [0]  # colonne x = 0
                threated_image[:, threated_image.shape[1] - 1] = [0]  # colonne y = (width - 1)
                cv2.imwrite(img_name + ".jpg", threated_image)
                # afficher l'image filtrée
                pixmap = QPixmap(img_name + ".jpg")
                self.label_21.setPixmap(pixmap)

        return threated_image

    def clamb_0_to_255(self, threated_image, filter_type):
        if (filter_type == "Mean"):
            for i in range(len(threated_image)):
                for j in range(len(threated_image[0])):
                    pixel_b = threated_image[i][j][0]
                    pixel_g = threated_image[i][j][1]
                    pixel_r = threated_image[i][j][2]
                    if (pixel_b < 0 and pixel_g < 0 and pixel_r < 0):
                        threated_image[i][j] = [0, 0, 0]
                        if (self.isJpg == False):  # image png
                            alpha = threated_image[i][j][3]
                            threated_image[i][j] = [0, 0, 0, alpha]
                        print('valeur inférieure à 0 trouvée')
                    if (pixel_b > 255 and pixel_g > 255 and pixel_r > 255):
                        threated_image[i][j] = [255, 255, 255]
                        if (self.isJpg == False):  # image png
                            alpha = threated_image[i][j][3]
                            threated_image[i][j] = [255, 255, 255, alpha]
                        print('valeur supérieure à 255 trouvée')
        else:
            for i in range(len(threated_image)):
                for j in range(len(threated_image)):
                    pixel = threated_image[i][j]
                    if (pixel < 0):
                        print('valeur inférieure à 0 trouvée')
                        threated_image[i][j] = 0
                    elif (pixel > 255):
                        print('valeur supérieure à 255 trouvée')
                        threated_image[i][j] = 255

        return threated_image

    def sobel_detection_contours(self, image, filter, type="sobel", verbose=False):
        new_image_x = self.convolution(image, filter, verbose)
        new_image_y = self.convolution(image, np.flip(filter.T, axis=0), verbose)

        gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

        if (type == "sobel_vert"):
            new_image_y *= 255.0 / new_image_y.max()
            gradient_magnitude = new_image_y
        elif (type == "sobel_horiz"):
            new_image_x *= 255.0 / new_image_x.max()
            gradient_magnitude = new_image_x
        else:
            gradient_magnitude *= 255.0 / gradient_magnitude.max()

        return gradient_magnitude

    def dnorm(self, x, mu, sd):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

    def gaussian_kernel(self, size, sigma=1, verbose=False):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = self.dnorm(kernel_1D[i], 0, sigma)
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

        kernel_2D *= 1.0 / kernel_2D.max()

        return kernel_2D

    def gaussian_blur(self, image, my_kernel, kernel_size=5, verbose=False, isCustom=False):
        kernel = self.gaussian_kernel(kernel_size, sigma=np.sqrt(kernel_size), verbose=verbose) \
            if not isCustom else my_kernel
        print(kernel)

        return self.convolution(image, kernel, average=True, verbose=verbose)

    def convolution(self, image, kernel, average=False, verbose=False):
        if len(image.shape) == 3:
            print("Found 3 Channels : {}".format(image.shape))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            print("Image Shape : {}".format(image.shape))

        print("Kernel Shape : {}".format(kernel.shape))

        image_row, image_col = image.shape
        kernel_row, kernel_col = kernel.shape

        output = np.zeros(image.shape)

        pad_height = int((kernel_row - 1) / 2)
        pad_width = int((kernel_col - 1) / 2)

        padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

        padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

        for row in range(image_row):
            for col in range(image_col):
                output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
                if average:
                    output[row, col] /= kernel.shape[0] * kernel.shape[1]

        print("Output Image size : {}".format(output.shape))

        return output

    def setupUi(self, Lab2_Window):
        Lab2_Window.setObjectName("Lab2_Window")
        # Lab2_Window.resize(832, 629)
        Lab2_Window.showMaximized()
        self.centralwidget = QtWidgets.QWidget(Lab2_Window)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame_3 = QtWidgets.QFrame(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMaximumSize(QtCore.QSize(16777215, 130))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_14 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_14.setObjectName("gridLayout_14")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_14.addItem(spacerItem, 0, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_14.addWidget(self.pushButton_2, 0, 3, 1, 1)
        self.frame_6 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setMinimumSize(QtCore.QSize(300, 100))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_2.addWidget(self.lineEdit_2, 0, 1, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_6.setInputMask("")
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout_2.addWidget(self.lineEdit_6, 1, 2, 1, 1)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout_2.addWidget(self.lineEdit_7, 2, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_2.addWidget(self.lineEdit, 0, 0, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout_2.addWidget(self.lineEdit_5, 1, 1, 1, 1)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.gridLayout_2.addWidget(self.lineEdit_9, 2, 2, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout_2.addWidget(self.lineEdit_3, 0, 2, 1, 1)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.gridLayout_2.addWidget(self.lineEdit_8, 2, 1, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.frame_6)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout_2.addWidget(self.lineEdit_4, 1, 0, 1, 1)
        self.gridLayout_14.addWidget(self.frame_6, 0, 2, 1, 1)
        self.frame_2 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.frame_2)
        self.formLayout_2.setFormAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_19 = QtWidgets.QLabel(self.frame_2)
        self.label_19.setObjectName("label_19")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.comboBox_7 = QtWidgets.QComboBox(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_7.sizePolicy().hasHeightForWidth())
        self.comboBox_7.setSizePolicy(sizePolicy)
        self.comboBox_7.setObjectName("comboBox_7")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.comboBox_7.addItem("")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_7)
        self.label_18 = QtWidgets.QLabel(self.frame_2)
        self.label_18.setObjectName("label_18")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.comboBox_6 = QtWidgets.QComboBox(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_6.sizePolicy().hasHeightForWidth())
        self.comboBox_6.setSizePolicy(sizePolicy)
        self.comboBox_6.setObjectName("comboBox_6")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.comboBox_6)
        self.comboBox_5 = QtWidgets.QComboBox(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_5.sizePolicy().hasHeightForWidth())
        self.comboBox_5.setSizePolicy(sizePolicy)
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_5)
        self.label_17 = QtWidgets.QLabel(self.frame_2)
        self.label_17.setObjectName("label_17")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.gridLayout_14.addWidget(self.frame_2, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_14.addItem(spacerItem1, 0, 4, 1, 1)
        self.gridLayout_3.addWidget(self.frame_3, 0, 0, 1, 1)
        self.frame_5 = QtWidgets.QFrame(self.tab)
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_4 = QtWidgets.QFrame(self.frame_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame_4)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.frame_4)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.verticalLayout.addWidget(self.frame_4)
        self.frame = QtWidgets.QFrame(self.frame_5)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.label_21 = QtWidgets.QLabel(self.frame)
        self.label_21.setFrameShape(QtWidgets.QFrame.Box)
        self.label_21.setText("")
        self.label_21.setObjectName("label_21")
        self.horizontalLayout_3.addWidget(self.label_21)
        self.verticalLayout.addWidget(self.frame)
        self.gridLayout_3.addWidget(self.frame_5, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.frame_20 = QtWidgets.QFrame(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_20.sizePolicy().hasHeightForWidth())
        self.frame_20.setSizePolicy(sizePolicy)
        self.frame_20.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_20.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_20.setObjectName("frame_20")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_20)
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem2, 0, 0, 1, 1)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.formLayout_4.setFormAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTop | QtCore.Qt.AlignTrailing)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_34 = QtWidgets.QLabel(self.frame_20)
        self.label_34.setObjectName("label_34")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_34)
        self.lineEdit_12 = QtWidgets.QLineEdit(self.frame_20)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_12.sizePolicy().hasHeightForWidth())
        self.lineEdit_12.setSizePolicy(sizePolicy)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_12)
        self.label_35 = QtWidgets.QLabel(self.frame_20)
        self.label_35.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_35.setObjectName("label_35")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_35)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.frame_20)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_13)
        self.label_36 = QtWidgets.QLabel(self.frame_20)
        self.label_36.setObjectName("label_36")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_36)
        self.lineEdit_14 = QtWidgets.QLineEdit(self.frame_20)
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_14)
        self.gridLayout_4.addLayout(self.formLayout_4, 0, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem3, 0, 2, 1, 1)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_20)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_10.addWidget(self.pushButton_3)
        self.gridLayout_4.addLayout(self.verticalLayout_10, 0, 3, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem4, 0, 4, 1, 1)
        self.verticalLayout_9.addWidget(self.frame_20)
        self.frame_17 = QtWidgets.QFrame(self.tab_2)
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_17)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.frame_18 = QtWidgets.QFrame(self.frame_17)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_18.sizePolicy().hasHeightForWidth())
        self.frame_18.setSizePolicy(sizePolicy)
        self.frame_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.frame_18)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.label_28 = QtWidgets.QLabel(self.frame_18)
        self.label_28.setAlignment(QtCore.Qt.AlignCenter)
        self.label_28.setObjectName("label_28")
        self.gridLayout_12.addWidget(self.label_28, 0, 0, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.frame_18)
        self.label_29.setAlignment(QtCore.Qt.AlignCenter)
        self.label_29.setObjectName("label_29")
        self.gridLayout_12.addWidget(self.label_29, 0, 1, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.frame_18)
        self.label_30.setAlignment(QtCore.Qt.AlignCenter)
        self.label_30.setObjectName("label_30")
        self.gridLayout_12.addWidget(self.label_30, 0, 2, 1, 1)
        self.verticalLayout_8.addWidget(self.frame_18)
        self.frame_19 = QtWidgets.QFrame(self.frame_17)
        self.frame_19.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_19.setObjectName("frame_19")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.frame_19)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.label_32 = QtWidgets.QLabel(self.frame_19)
        self.label_32.setFrameShape(QtWidgets.QFrame.Box)
        self.label_32.setText("")
        self.label_32.setObjectName("label_32")
        self.gridLayout_13.addWidget(self.label_32, 0, 1, 1, 1)
        self.label_31 = QtWidgets.QLabel(self.frame_19)
        self.label_31.setFrameShape(QtWidgets.QFrame.Box)
        self.label_31.setText("")
        self.label_31.setObjectName("label_31")
        self.gridLayout_13.addWidget(self.label_31, 0, 0, 1, 1)
        self.label_33 = QtWidgets.QLabel(self.frame_19)
        self.label_33.setFrameShape(QtWidgets.QFrame.Box)
        self.label_33.setText("")
        self.label_33.setObjectName("label_33")
        self.gridLayout_13.addWidget(self.label_33, 0, 2, 1, 1)
        self.verticalLayout_8.addWidget(self.frame_19)
        self.verticalLayout_9.addWidget(self.frame_17)
        self.frame_13 = QtWidgets.QFrame(self.tab_2)
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_13)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.frame_15 = QtWidgets.QFrame(self.frame_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_15.sizePolicy().hasHeightForWidth())
        self.frame_15.setSizePolicy(sizePolicy)
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.frame_15)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_16 = QtWidgets.QLabel(self.frame_15)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.gridLayout_10.addWidget(self.label_16, 0, 0, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.frame_15)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.gridLayout_10.addWidget(self.label_23, 0, 1, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.frame_15)
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.gridLayout_10.addWidget(self.label_24, 0, 2, 1, 1)
        self.verticalLayout_7.addWidget(self.frame_15)
        self.frame_16 = QtWidgets.QFrame(self.frame_13)
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.frame_16)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.label_25 = QtWidgets.QLabel(self.frame_16)
        self.label_25.setFrameShape(QtWidgets.QFrame.Box)
        self.label_25.setText("")
        self.label_25.setObjectName("label_25")
        self.gridLayout_11.addWidget(self.label_25, 0, 0, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.frame_16)
        self.label_26.setFrameShape(QtWidgets.QFrame.Box)
        self.label_26.setText("")
        self.label_26.setObjectName("label_26")
        self.gridLayout_11.addWidget(self.label_26, 0, 1, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.frame_16)
        self.label_27.setFrameShape(QtWidgets.QFrame.Box)
        self.label_27.setText("")
        self.label_27.setObjectName("label_27")
        self.gridLayout_11.addWidget(self.label_27, 0, 2, 1, 1)
        self.verticalLayout_7.addWidget(self.frame_16)
        self.verticalLayout_9.addWidget(self.frame_13)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_14 = QtWidgets.QFrame(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_14.sizePolicy().hasHeightForWidth())
        self.frame_14.setSizePolicy(sizePolicy)
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.frame_14)
        self.gridLayout_9.setObjectName("gridLayout_9")
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem5, 0, 0, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem6, 0, 3, 1, 1)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setFormAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft)
        self.formLayout.setObjectName("formLayout")
        self.label_20 = QtWidgets.QLabel(self.frame_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy)
        self.label_20.setMinimumSize(QtCore.QSize(145, 0))
        self.label_20.setObjectName("label_20")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.frame_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_10.sizePolicy().hasHeightForWidth())
        self.lineEdit_10.setSizePolicy(sizePolicy)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_10)
        self.verticalLayout_5.addLayout(self.formLayout)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_22 = QtWidgets.QLabel(self.frame_14)
        self.label_22.setObjectName("label_22")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_22)
        self.lineEdit_11 = QtWidgets.QLineEdit(self.frame_14)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_11)
        self.verticalLayout_5.addLayout(self.formLayout_3)
        self.gridLayout_9.addLayout(self.verticalLayout_5, 0, 2, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_9.addItem(spacerItem7, 0, 5, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton = QtWidgets.QPushButton(self.frame_14)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_4.addWidget(self.pushButton)
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_14)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_4.addWidget(self.pushButton_4)
        self.gridLayout_9.addLayout(self.verticalLayout_4, 0, 4, 1, 1)
        self.verticalLayout_6.addWidget(self.frame_14)
        self.frame_7 = QtWidgets.QFrame(self.tab_3)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_9 = QtWidgets.QFrame(self.frame_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_9.sizePolicy().hasHeightForWidth())
        self.frame_9.setSizePolicy(sizePolicy)
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.frame_9)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_4 = QtWidgets.QLabel(self.frame_9)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_5.addWidget(self.label_4, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame_9)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_5.addWidget(self.label_5, 0, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.frame_9)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout_5.addWidget(self.label_6, 0, 2, 1, 1)
        self.verticalLayout_2.addWidget(self.frame_9)
        self.frame_10 = QtWidgets.QFrame(self.frame_7)
        self.frame_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.frame_10)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.label_7 = QtWidgets.QLabel(self.frame_10)
        self.label_7.setFrameShape(QtWidgets.QFrame.Box)
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.gridLayout_8.addWidget(self.label_7, 0, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.frame_10)
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.gridLayout_8.addWidget(self.label_8, 0, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.frame_10)
        self.label_9.setFrameShape(QtWidgets.QFrame.Box)
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.gridLayout_8.addWidget(self.label_9, 0, 2, 1, 1)
        self.verticalLayout_2.addWidget(self.frame_10)
        self.verticalLayout_6.addWidget(self.frame_7)
        self.frame_8 = QtWidgets.QFrame(self.tab_3)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_8)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_12 = QtWidgets.QFrame(self.frame_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_12.sizePolicy().hasHeightForWidth())
        self.frame_12.setSizePolicy(sizePolicy)
        self.frame_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.frame_12)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_10 = QtWidgets.QLabel(self.frame_12)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_6.addWidget(self.label_10, 0, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.frame_12)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout_6.addWidget(self.label_11, 0, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.frame_12)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.gridLayout_6.addWidget(self.label_12, 0, 2, 1, 1)
        self.verticalLayout_3.addWidget(self.frame_12)
        self.frame_11 = QtWidgets.QFrame(self.frame_8)
        self.frame_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.frame_11)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_13 = QtWidgets.QLabel(self.frame_11)
        self.label_13.setFrameShape(QtWidgets.QFrame.Box)
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.gridLayout_7.addWidget(self.label_13, 0, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.frame_11)
        self.label_14.setFrameShape(QtWidgets.QFrame.Box)
        self.label_14.setText("")
        self.label_14.setObjectName("label_14")
        self.gridLayout_7.addWidget(self.label_14, 0, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.frame_11)
        self.label_15.setFrameShape(QtWidgets.QFrame.Box)
        self.label_15.setText("")
        self.label_15.setObjectName("label_15")
        self.gridLayout_7.addWidget(self.label_15, 0, 2, 1, 1)
        self.verticalLayout_3.addWidget(self.frame_11)
        self.verticalLayout_6.addWidget(self.frame_8)
        self.tabWidget.addTab(self.tab_3, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        Lab2_Window.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Lab2_Window)
        self.statusbar.setObjectName("statusbar")
        Lab2_Window.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(Lab2_Window)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 832, 21))
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        self.menuFilter = QtWidgets.QMenu(self.menuBar)
        self.menuFilter.setObjectName("menuFilter")
        Lab2_Window.setMenuBar(self.menuBar)
        self.actionExit = QtWidgets.QAction(Lab2_Window)
        self.actionExit.setObjectName("actionExit")
        self.actionAdd_Image = QtWidgets.QAction(Lab2_Window)
        self.actionAdd_Image.setObjectName("actionAdd_Image")
        self.actionApply_Filter = QtWidgets.QAction(Lab2_Window)
        self.actionApply_Filter.setObjectName("actionApply_Filter")
        self.menuFile.addAction(self.actionExit)
        self.menuFilter.addAction(self.actionAdd_Image)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuFilter.menuAction())

        self.label_3.setScaledContents(True)
        self.label_21.setScaledContents(True)
        self.lineEdit.setValidator(QDoubleValidator(-999999, 999999, 8))
        self.lineEdit_2.setValidator(QDoubleValidator(-999999, 999999, 8))
        self.lineEdit_3.setValidator(QDoubleValidator(-999999, 999999, 8))
        self.lineEdit_4.setValidator(QDoubleValidator(-999999, 999999, 8))
        self.lineEdit_5.setValidator(QDoubleValidator(-999999, 999999, 8))
        self.lineEdit_6.setValidator(QDoubleValidator(-999999, 999999, 8))
        self.lineEdit_7.setValidator(QDoubleValidator(-999999, 999999, 8))
        self.lineEdit_8.setValidator(QDoubleValidator(-999999, 999999, 8))
        self.lineEdit_9.setValidator(QDoubleValidator(-999999, 999999, 8))
        self.pushButton_2.clicked.connect(self.applyFilter)
        self.pushButton.clicked.connect(self.applyIdealLowPassFilter)
        self.pushButton_4.clicked.connect(self.applyButterworthFilter)
        self.comboBox_5.currentIndexChanged.connect(self.filterChanged)
        self.actionAdd_Image.triggered.connect(self.openImage)

        self.retranslateUi(Lab2_Window)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Lab2_Window)

    def retranslateUi(self, Lab2_Window):
        _translate = QtCore.QCoreApplication.translate
        Lab2_Window.setWindowTitle(_translate("Lab2_Window", "Lab2_Window"))
        self.pushButton_2.setText(_translate("Lab2_Window", "Apply Filter"))
        self.label_19.setText(_translate("Lab2_Window", "Handling Borders"))
        self.comboBox_7.setItemText(0, _translate("Lab2_Window", "0"))
        self.comboBox_7.setItemText(1, _translate("Lab2_Window", "None"))
        self.comboBox_7.setItemText(2, _translate("Lab2_Window", "Copy"))
        self.comboBox_7.setItemText(3, _translate("Lab2_Window", "Mirror"))
        self.comboBox_7.setItemText(4, _translate("Lab2_Window", "Circular"))
        self.label_18.setText(_translate("Lab2_Window", "Range"))
        self.comboBox_6.setItemText(0, _translate("Lab2_Window", "Clamp 0 ... 255"))
        self.comboBox_6.setItemText(1, _translate("Lab2_Window", "Abs and normalize to 255"))
        self.comboBox_6.setItemText(2, _translate("Lab2_Window", "Abs and normalize 0 to 255"))
        self.comboBox_6.setItemText(3, _translate("Lab2_Window", "Normalize 0 to 255"))
        self.comboBox_5.setItemText(0, _translate("Lab2_Window", "Costum"))
        self.comboBox_5.setItemText(1, _translate("Lab2_Window", "Mean"))
        self.comboBox_5.setItemText(2, _translate("Lab2_Window", "Gaussian"))
        self.comboBox_5.setItemText(3, _translate("Lab2_Window", "4 - Neighbour Laplacian"))
        self.comboBox_5.setItemText(4, _translate("Lab2_Window", "8 - Neighbour Laplacian"))
        self.comboBox_5.setItemText(5, _translate("Lab2_Window", "Sobel Horiz"))
        self.comboBox_5.setItemText(6, _translate("Lab2_Window", "Sobel Vert"))
        self.comboBox_5.setItemText(7, _translate("Lab2_Window", "Sobel"))
        self.label_17.setText(_translate("Lab2_Window", "Filter Type"))
        self.label.setText(_translate("Lab2_Window", "Original Image"))
        self.label_2.setText(_translate("Lab2_Window", "Filtered Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Lab2_Window", "Spatial Filters"))
        self.label_34.setText(_translate("Lab2_Window", "Gaussian Filter Size"))
        self.label_35.setText(_translate("Lab2_Window", "Min Threshold"))
        self.label_36.setText(_translate("Lab2_Window", "Max Threshold"))
        self.pushButton_3.setText(_translate("Lab2_Window", "Apply Filter"))
        self.label_28.setText(_translate("Lab2_Window", "Original Image"))
        self.label_29.setText(_translate("Lab2_Window", "Gradient X"))
        self.label_30.setText(_translate("Lab2_Window", "Local Maxima"))
        self.label_16.setText(_translate("Lab2_Window", "Smoothed Image"))
        self.label_23.setText(_translate("Lab2_Window", "Gradient Y"))
        self.label_24.setText(_translate("Lab2_Window", "Final Contour Image"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Lab2_Window", "Canny Algorithm"))
        self.label_20.setText(_translate("Lab2_Window", "N parameter for Low-Pass"))
        self.label_22.setText(_translate("Lab2_Window", "N parameter for Butterworth  "))
        self.pushButton.setText(_translate("Lab2_Window", "Apply Ideal Low-Pass Filter"))
        self.pushButton_4.setText(_translate("Lab2_Window", "Apply Butterworth Filter"))
        self.label_4.setText(_translate("Lab2_Window", "Original Image"))
        self.label_5.setText(_translate("Lab2_Window", " Ideal Low-Pass reconstructed Image 1"))
        self.label_6.setText(_translate("Lab2_Window", "Low-Pass Butterworth reconstructed Image 1"))
        self.label_10.setText(_translate("Lab2_Window", "Original Spectrum"))
        self.label_11.setText(_translate("Lab2_Window", "Ideal Low-Pass Spectrum 1"))
        self.label_12.setText(_translate("Lab2_Window", "Low-Pass Butterworth Spectrum 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Lab2_Window", "Frequency Filters"))
        self.menuFile.setTitle(_translate("Lab2_Window", "File"))
        self.menuFilter.setTitle(_translate("Lab2_Window", "Add"))
        self.actionExit.setText(_translate("Lab2_Window", "Exit"))
        self.actionAdd_Image.setText(_translate("Lab2_Window", "Add Image"))
        self.actionApply_Filter.setText(_translate("Lab2_Window", "Apply Filter"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Lab2_Window = QtWidgets.QMainWindow()
    ui = Ui_Lab2_Window()
    ui.setupUi(Lab2_Window)
    Lab2_Window.show()
    sys.exit(app.exec_())
