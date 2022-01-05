import numpy as np
import cv2
import matplotlib.pyplot as plt


def save_img(name, img, flag =None):
    if(flag != None):
        cv2.imwrite(name, cv2.cvtColor(img.astype(np.uint8), flag))
    else:
        cv2.imwrite(name, img.astype(np.uint8))


def filtering(img, filter, type_filter):

    filtered_img = np.zeros_like(img, dtype='float32')

    for i in range(img.shape[2]):
        img_chan = img[:,:,i] # gray-scale image


        f = np.fft.fft2(img_chan.astype(np.float32))
        f_shifted = np.fft.fftshift(f)

        # if type_filter == True:
        #     filter[np.abs(f_complex) > np.mean(np.abs(f_complex))] = 1
        # else:
        #     filter[np.abs(f_complex) > np.mean(np.abs(f_complex))] = 0

        f_filtered = filter * f_shifted

        f_filtered_shifted = np.fft.ifftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
        filtered_img[:,:,i] = np.real(inv_img)

    return filtered_img



def get_filter_image(img, lpf = True):

    r = 5e4 # how narrower the window is
    ham1 = np.hamming(img.shape[0])[:,None] # 1D hamming
    ham2 = np.hamming(img.shape[1])[:,None] # 1D hamming
    ham2d = np.sqrt(np.dot(ham1, ham2.T)) ** r # expand to 2D hamming
    hm = 2**(-1/(2*r))
    bound = 50 - np.arcsin(hm)/np.pi*100
    print("bound", bound, " %")

    if(lpf == True):
        ham2d = ham2d
    else:
        ham2d = np.ones_like(ham2d)  - ham2d



    filtered_img = filtering(img, ham2d, lpf)

    return filtered_img

