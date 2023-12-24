import numpy as np
import scipy 
import os
import cv2
import matplotlib.pyplot as plt
import math
from scipy import ndimage
from ref_framework import *

def test_Harris(path = './Problem2Images/',filename = '3_1.jpg'):
    # read the image using 
    img = cv2.imread(path+filename)
    # convert the image to grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calculate harris response
    R = harris_response(img_grey, 0.04, 3)
    corners = corner_selection(R, 0.01,10)
    # plot the corners on the image and save it without showing axis
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    for corner in corners:
        plt.plot(corner[1], corner[0], 'r.', markersize=3)
    plt.axis('off')
    plt.savefig(path+'harris_'+filename,dpi=300)
    plt.close()

def test_HoG(path = './Problem2Images/',filename = '1_2.jpg'):
    # read the image using 
    img = cv2.imread(path+filename)
    # convert the image to grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calculate HoG
    R = harris_response(img_grey, 0.04, 3)
    corners = corner_selection(R, 0.01,10)
    HoG = histogram_of_gradients(img_grey,pix=corners)
    # plot the HoG on the image and save it without showing axis
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(HoG.shape, len(corners))
    plt.imshow(img)
    for idx,corner in enumerate(corners):
        plt.plot(corner[1], corner[0], 'r.', markersize=3)
        for idx1,hist in enumerate(HoG[idx]):
            plt.arrow(corner[1], corner[0],50*hist*math.cos(idx1*math.pi/4),50*hist*math.sin(idx1*math.pi/4),color='r',width=0.1)
    plt.axis('off')
    plt.savefig('hog_'+filename,dpi=300)

def test_feature_matching(path='./Problem2Images/',filename1='1_1.jpg',filename2 = '1_2.jpg'):
    img1 = cv2.imread(path+filename1)
    img2 = cv2.imread(path+filename2)
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    pix_1,pix_2 = feature_matching(img1_grey,img2_grey)
    # print(len(pix_1),len(pix_2))
    # print(pix_2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    best_homo_matrix = align_pair(pix_1,pix_2)
    # print(best_homo_matrix)
    pixels_1 = homo_coordinates(pix_1)
    pix_21 = Cartesian_coodinates((np.matmul( best_homo_matrix,np.array(pixels_1).T)).T)
    # pix_21 = Cartesian_coodinates((np.array(pixels_1).dot(best_homo_matrix)))
    # print(np.array(pix_21)-np.array(pix_2))
    distances = np.linalg.norm(pix_21 - np.array(pix_2), axis=1)
    # find the smallest 10 distance in distances and calculate the mean
    smallest = np.sort(distances,axis = 0)[:int(0.8*len(distances))]
    distance_mean = np.mean(smallest)
    print(f'the mean distance between {path+filename1} and {filename2} is {distance_mean} after feature_matching')

def test_stiched_img(path='./Problem2Images/',filename1='3_1.jpg',filename2 = '3_2.jpg'):
    img1 = cv2.imread(path+filename1)
    img2 = cv2.imread(path+filename2)
    # print(img1.shape)
    # plt.imshow(img1)
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    pix_1,pix_2 = feature_matching(img1_grey,img2_grey)
    best_homo_matrix = align_pair(pix_1,pix_2)
    # print(f'best_homo_matrix=\n{best_homo_matrix}')
    stitched_img = stitch_blend(img1,img2,best_homo_matrix)
    stitched_img = cv2.cvtColor(stitched_img.astype(np.uint8),cv2.COLOR_BGR2RGB)
    print(f'{path+filename1} has been stitched!')
    # print(stitched_img.shape)
    plt.imshow(stitched_img)
    plt.savefig(path+filename1.split('.')[0]+' '+'stitch.jpg',dpi=300)
    plt.close()

def test_panorama(file_list,path):
    imgs = []
    for file in file_list[:4]:
        img = cv2.imread(path+file)
        imgs.append(img)
    result = generate_panorama(imgs)
    result = cv2.cvtColor(result.astype(np.uint8),cv2.COLOR_BGR2RGB)
    plt.imshow(result)
    plt.savefig(file_list[0].split('.')[0]+'panorama.jpg')
    plt.close()
    # plt.show()

if __name__ == '__main__':
    # Harris_files = ['1_1.jpg','2_1.jpg','3_1.JPG']
    # for filename in Harris_files:
    #     test_Harris(filename = filename)
    # # test_HoG(filename = '1_1.jpg')
    # # HoG_files = ['2_1.jpg','2_2.jpg']
    # # for filename in HoG_files:
    # #     test_HoG(filename = filename)
    # feature_matching_files = [('1_1.jpg','1_2.jpg'),('2_1.jpg','2_2.jpg'),('3_1.JPG','3_2.JPG')]
    # for file1,file2 in feature_matching_files:
    #     test_feature_matching(filename1 = file1,filename2 = file2)
    
    # stitch_files = [('1_1.jpg','1_2.jpg'),('2_1.jpg','2_2.jpg'),('3_1.JPG','3_2.JPG')]
    # for file1,file2 in stitch_files:
    #     test_stiched_img(filename1 = file1,filename2 = file2)

    path = './Problem2Images/panoramas/grail/'
    # path = './Problem2Images/panoramas/parrington/'
    panorama_file_list = os.listdir(path)
    test_panorama(panorama_file_list,path)
    