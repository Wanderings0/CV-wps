# Name: Siyuan Yin 2100017768
import numpy as np
import cv2
import matplotlib.pyplot as plt

# cross_correlation_2d for 2D cross correlation
def cross_correlation_2d(img,kenerl):
    # get the size of image and kenerl
    img_row, img_col = img.shape
    kenerl_row, kenerl_col = kenerl.shape
    # get the padding size
    pad_row = int((kenerl_row-1)/2)
    pad_col = int((kenerl_col-1)/2)
    # padding the image
    img_pad = np.zeros((img_row+pad_row*2,img_col+pad_col*2))
    img_pad[pad_row:pad_row+img_row,pad_col:pad_col+img_col] = img
    # get the result
    result = np.zeros((img_row,img_col))
    for i in range(img_row):
        for j in range(img_col):
            result[i,j] = np.sum(img_pad[i:i+kenerl_row,j:j+kenerl_col]*kenerl)
    return result

# convolve_2d for 2D convolution using cross_correlation_2d
def convolve_2d(img,kenerl):
    return cross_correlation_2d(img,kenerl[::-1,::-1])

# gaussian_blur_kernel_2d for 2D gaussian blur kernel
def gaussian_blur_kernel_2d(sigma,kenerl_size):
    # get the size of kenerl
    kenerl_row, kenerl_col = kenerl_size
    # get the center of kenerl
    center_row = int((kenerl_row-1)/2)
    center_col = int((kenerl_col-1)/2)
    # get the gaussian kenerl
    gaussian_kenerl = np.zeros((kenerl_row,kenerl_col))
    for i in range(kenerl_row):
        for j in range(kenerl_col):
            gaussian_kenerl[i,j] = np.exp(-((i-center_row)**2+(j-center_col)**2)/(2*sigma**2))
    # normalize the kenerl
    gaussian_kenerl = gaussian_kenerl/np.sum(gaussian_kenerl)
    return gaussian_kenerl

# gaussian_blur_2d for 2D gaussian blur using gaussian_blur_kernel_2d
def low_pass(img,sigma,kenerl_size):
    gaussian_kenerl = gaussian_blur_kernel_2d(sigma,kenerl_size)
    return convolve_2d(img,gaussian_kenerl)

# image subsample
def image_subsampling(img,s=2):
    # get the size of image
    img_row, img_col = img.shape
    # subsample the image
    img_subsample = np.zeros((int(img_row/2),int(img_col/2)))
    for i in range(int(img_row/2)):
        for j in range(int(img_col/2)):
            img_subsample[i,j] = img[i*2,j*2]
    return img_subsample

#gaussian_pyramid
def gaussian_pyramid(img,sigma,kenerl_size,level):
    # get the size of image
    img_row, img_col = img.shape
    # get the gaussian pyramid
    gaussian_pyramid = []
    for i in range(level):
        # get the gaussian blur image
        gaussian_img = low_pass(img,sigma,kenerl_size)
        # downsample the image
        img = image_subsampling(gaussian_img)
        # add the image to gaussian pyramid
        gaussian_pyramid.append(gaussian_img)
    return gaussian_pyramid

# main function using gaussian_pyramid
def main():
    # read the image
    img = cv2.imread('Lena.png',0)
    # img = cv2.imread('VanGogh.png',0)

    # get the gaussian pyramid
    gaussian = gaussian_pyramid(img,1,(3,3),4)
    # gaussian1.png is original image with gaussian blur
    # gaussian2.png, gaussian3.png, gaussian4.png are the downsampled images
    for i in range(len(gaussian)):
        cv2.imwrite('gaussian'+str(i+1)+'.png',gaussian[i])
    

if __name__ == '__main__':
    main()