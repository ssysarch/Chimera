#  Author: fengping su
#  date: 2023-8-23
#  All rights reserved.

import cv2
import numpy as np
import pywt
import torch


# implementation of filter bank preprocess module using DCT
class DCT(torch.nn.Module):
    def __init__(self, k=10):
        super(DCT, self).__init__()
        self.k = k

    def forward(self, pil_img):
        return self._filter_bank_preprocess(pil_img, self.k)

    @staticmethod
    def _filter_bank_preprocess(pil_img, k: int = 10):
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY).astype(np.float32)
        img_dct = cv2.dct(img)
        img_dct_f = np.abs(img_dct)

        low_freq = img_dct_f > 2 * k
        mid_freq = (img_dct_f >= k) * (img_dct_f < 2 * k)
        high_freq = img_dct_f < k

        img_dct_low = cv2.idct(img_dct * low_freq)[None, :, :]
        img_dct_mid = cv2.idct(img_dct * mid_freq)[None, :, :]
        img_dct_high = cv2.idct(img_dct * high_freq)[None, :, :]
        img = np.concatenate((img_dct_low, img_dct_mid, img_dct_high), axis=0)
        return torch.from_numpy(img).contiguous() / 255.0


class DWT(torch.nn.Module):
    def __init__(self, wavelet="haar"):
        super(DWT, self).__init__()
        self.wavelet = wavelet

    def forward(self, pil_img):
        return self._filter_bank_preprocess(pil_img, self.wavelet)

    @staticmethod
    def _filter_bank_preprocess(pil_img, wavelet):
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY).astype(np.float32)
        coeffs = pywt.dwt2(img, wavelet)
        LL, (LH, HL, HH) = coeffs

        img = np.stack((LH, HL, HH), axis=0)
        return torch.from_numpy(img).contiguous() / 255.0


class FFT(torch.nn.Module):
    def __init__(self):
        super(FFT, self).__init__()

    def forward(self, pil_img):
        return self._filter_bank_preprocess(pil_img)

    @staticmethod
    def _filter_bank_preprocess(pil_img):
        gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY).astype(np.float32)
        f = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1e-5)
        img = np.stack((magnitude, magnitude, magnitude), axis=0)

        return torch.from_numpy(img).contiguous() / 255.0


class Laplacian(torch.nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()

    def forward(self, pil_img):
        return self._filter_bank_preprocess(pil_img)

    @staticmethod
    def _filter_bank_preprocess(pil_img):
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY).astype(np.float32)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)  # output shape: size x size
        abs_laplacian = cv2.convertScaleAbs(laplacian)
        img = np.stack((abs_laplacian, abs_laplacian, abs_laplacian), axis=0)

        return torch.from_numpy(img).contiguous() / 255.0
