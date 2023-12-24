# This is a raw framework for image stitching using Harris corner detection.
# For libraries you can use modules in numpy, scipy, cv2, os, etc.
import numpy as np
import scipy 
import cv2
import matplotlib.pyplot as plt
import math
from scipy import ndimage
from scipy import signal 
import random


def gradient_x(img):
    # convert img to grayscale
    # should we use int type to calclate gradient?
    # should we conduct some pre-processing to remove noise? which kernel should we apply?
    # which kernel should we choose to calculate gradient_x?
    # TODO
    '''
    input: grayscale img, np.array((W,H))
    output: grad_x, np.array((W,H))
    '''
    gaussian_kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
    # conduct gaussian blur
    img = signal.convolve2d(img, gaussian_kernel, mode='same', boundary='symm')
    sobel_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # calculate gradient_x
    grad_x = signal.convolve2d(img, sobel_kernel, mode='same', boundary='symm')
    return grad_x

def gradient_y(img):
    # TODO
    '''
    input: grayscale img, np.array((W,H))
    output: grad_y, np.array((W,H))
    '''
    gaussian_kernel = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
    # conduct gaussian blur
    img = signal.convolve2d(img, gaussian_kernel, mode='same', boundary='symm')
    sobel_kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    # calculate gradient_y
    grad_y = signal.convolve2d(img, sobel_kernel, mode='same', boundary='symm')
    return grad_y

def harris_response(img, alpha, win_size):
    # In this function you are going to claculate harris response R.
    # Please refer to 04_Feature_Detection.pdf page 29 for details. 
    # You have to discover how to calculate det(M) and trace(M), and
    # remember to smooth the gradients. 
    # Avoid using too much "for" loops to speed up.
    # TODO
    I_x = gradient_x(img)
    I_y = gradient_y(img)
    # calculate Ixx, Iyy, Ixy
    Ixx = I_x * I_x
    Iyy = I_y * I_y
    Ixy = I_x * I_y
    # calculate M
    M = np.zeros((img.shape[0], img.shape[1], 2, 2))
    M[:,:,0,0] = cv2.GaussianBlur(Ixx, (win_size,win_size), 1,borderType=cv2.BORDER_REFLECT)
    M[:,:,0,1] = cv2.GaussianBlur(Ixy, (win_size,win_size), 1,borderType=cv2.BORDER_REFLECT)
    M[:,:,1,0] = cv2.GaussianBlur(Ixy, (win_size,win_size), 1,borderType=cv2.BORDER_REFLECT)
    M[:,:,1,1] = cv2.GaussianBlur(Iyy, (win_size,win_size), 1,borderType=cv2.BORDER_REFLECT)

    # # calculate R
    det_M = np.linalg.det(M)
    trace_M = np.trace(M, axis1=2, axis2=3)
    R = det_M - alpha * trace_M * trace_M
    # scale the R into -1 to 1
    R = R / np.max(R)
    return R

def corner_selection(R, thresh, min_dist):
    # non-maximal suppression for R to get R_selection and transform selected corners to list of tuples
    # hint: 
    #   use ndimage.maximum_filter()  to achieve non-maximum suppression
    #   set those which aren’t **local maximum** to zero.
    ''' Input :
    R(np.array(W,H)): Pixel-wise Harris response
    thresh(float): Threshold for R for selecting corners
    min_dist(int): Minimum distance of two nearby corners
    Output:
    pixels(list:N): A list of list containing pixels selected as corners
    '''
    R_selection = np.zeros(R.shape)
    # TODO
    # non-maximal suppression
    R_max = ndimage.maximum_filter(R, size=min_dist)
    R_selection[R==R_max] = R[R==R_max]
    # select corners
    pixels = []
    for i in range(R_selection.shape[0]):
        for j in range(R_selection.shape[1]):
            if R_selection[i,j] > thresh:
                pixels.append([i,j])
    return pixels

def histogram_of_gradients(img, pix,norm=1):
    '''
    Input:
    img(np.array(W,H)): Grayscale image
    pix(list:N): A list of tuples contain N indices of pixels seleced as corners
    Output:
    features(np.array(N, L)):A list of L-dimensional feature vectors corresponding to N corners 
    '''
    # Hint: 
    #   1. grad_x & grad_y
    #   2. grad_dir by arctan function
    #   3. for each interest point, choose m*m blocks with each consists of m*m pixels
    #   4. I divide the region into n directions (maybe 8).
    #   5. For each blocks, calculate the number of derivatives in those directions and normalize the Histogram. 
    #   6. After that, select the prominent gradient and take it as principle orientation.
    #   7. Then rotate it’s neighbor to fit principle orientation and calculate the histogram again.     
    # TODO

    # calculate gradient
    grad_x = gradient_x(img)
    grad_y = gradient_y(img)
    # calculate gradient direction
    grad_dir = np.arctan2(grad_y, grad_x) # -pi to pi
    # calculate gradient magnitude
    grad_mag = np.sqrt(grad_x*grad_x + grad_y*grad_y)
    m = 6
    n = 8
    # biases = [[-1,-1],[-1,2],[2,-1],[2,2]]
    biases = [[-2,-2],[-2,3],[3,-2],[3,3]]
    features = []
    for x, y in pix:
        hist = []
        for bias in biases:
            x=x+bias[0]
            y=y+bias[1]
            half_size = m // 2
            window = grad_dir[x - half_size:x + half_size, y - half_size:y + half_size]
            weight = grad_mag[x - half_size:x + half_size, y - half_size:y + half_size]
            # print(window.shape)
            hist_block = np.ones(n)
           
            for i in range(window.shape[0]):
                for j in range(window.shape[1]):
                    angle = window[i, j]
                    
                    if angle==np.pi:
                        angle = -np.pi
                    bin_index = int((angle + np.pi) * n / (2 * np.pi))
                    hist_block[bin_index] += weight[i,j]
            hist.append(hist_block)
        # flatten the hist into 1-dim
        hist = np.array(hist).flatten()
        
        prominent_bin = np.argmax(hist)

        
        rotated_hist = np.roll(hist, n*4 - prominent_bin)

        # Normalize the histogram
        norm_hist = rotated_hist / np.sum(rotated_hist)
        hist = hist/np.sum(hist)

        if norm:
            features.append(norm_hist)
        else:
            features.append(hist)

    return np.array(features)

                
def feature_matching(img_1, img_2):
    # align two images using \verb|harris_response|, \verb|corner_selection|,
    # \verb|histogram_of_gradients|
    # hint: calculate distance by scipy.spacial.distance.cdist (using HoG features, euclidean will work well)
    '''Input:
    img_1 img_2(np.array(W,H)): Grayscale images
    Output:
    pix_1,pix_2: A list of tuples(x,y) that contains indices of pixels seleted as corners
                pix_1[i] in img_1 mathches pix_2[i] in img_2
    '''
    # TODO
    # calculate R
    R_1 = harris_response(img_1,0.04,3)
    R_2 = harris_response(img_2,0.04,3)
    # select corners
    pix_1 = corner_selection(R_1, 0.01, 10)
    pix_2 = corner_selection(R_2, 0.01, 10)
    # calculate HoG features
    features_1 = histogram_of_gradients(img_1, pix_1)
    features_2 = histogram_of_gradients(img_2, pix_2)
    # calculate distance
    dist = scipy.spatial.distance.cdist(features_1, features_2, metric='euclidean')
    # convert NAN to 1
    dist[np.isnan(dist)] = 1
    pixel_1 = []
    pixel_2 = []
    for i in range(len(pix_1)):
        j = np.argmin(dist[i,:])
        if i==np.argmin(dist[:,j]) and dist[i,j]<0.1:
            pixel_1.append(pix_1[i])
            pixel_2.append(pix_2[j])
    return pixel_1, pixel_2

def homo_coordinates(pixels):
    return [[x,y,1] for x,y in pixels]
def Cartesian_coodinates(pixels):
    return [[x/z,y/z] for x,y,z in pixels]

def compute_homography(pixels_1, pixels_2):
    # compute the best-fit homography using the Singular Value Decomposition (SVD)
    # homography matrix is a (3,3) matrix consisting rotation, translation and projection information.
    # consider how to form matrix A for U, S, V = np.linalg.svd((np.transpose(A)).dot(A))
    # homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    '''input:
    pixels_1,pixels_2: A list of lists[x,y]
    pixels_1 in img1 and pixels_2 in img2 are matching points 
    '''
    # TODO
    if len(pixels_1) != len(pixels_2) or len(pixels_1) < 4:
        raise ValueError("Both input lists must have the same number of points, and at least 4 points are required.")
    A = []
    for i in range(len(pixels_1)):
        x1, y1 = pixels_1[i]
        x2, y2 = pixels_2[i]
        A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])

    # Apply SVD to A
    A = -np.array(A)
    U, S, V = np.linalg.svd(A)

    # The homography matrix is the last column of V.
    homo_matrix = np.reshape(V[np.argmin(S)], (3, 3))
    homo_matrix = homo_matrix
    return homo_matrix

def align_pair(pixels_1, pixels_2):
    # utilize \verb|homo_coordinates| for homogeneous pixels
    # and \verb|compute_homography| to calulate homo_matrix
    # implement RANSAC to compute the optimal alignment.
    # you can refer to implementations online.
    num_iteration = 1000 # number of RANSAC iterations
    inlier_threshold = 2
    num_sample = 4
    best_inliers = []
    best_homo_matrix = None
    for _ in range(num_iteration):
        sample_indices = random.sample(range(len(pixels_1)),num_sample)
        sample_pixels_1 = [pixels_1[i] for i in sample_indices]
        sample_pixels_2 = [pixels_2[i] for i in sample_indices]

        homo_matrix = compute_homography(sample_pixels_1,sample_pixels_2)
        transformed_pixels_1 = Cartesian_coodinates((np.matmul(homo_matrix,np.array(homo_coordinates(pixels_1)).T)).T)

        # transformed_pixels_1 = cv2.perspectiveTransform(np.array([pixels_1], dtype=np.float32), homo_matrix)
        distances = np.linalg.norm(transformed_pixels_1 - np.array(pixels_2), axis=1)

        inliers = np.sum(distances<inlier_threshold)
        if inliers > len(best_inliers):
            best_inliers = np.where(distances<inlier_threshold)[0]
            best_homo_matrix = homo_matrix
    # print(best_inliers)
    best_inlier_pixels_1 = [pixels_1[i] for i in best_inliers]
    best_inlier_pixels_2 = [pixels_2[i] for i in best_inliers]
    best_homo_matrix = compute_homography(best_inlier_pixels_1, best_inlier_pixels_2)

    return best_homo_matrix

# def stitch_blend(img_1, img_2, best_homo):
#     # hint: 
#     # First, project four corner pixels with estimated homo-matrix
#     # and converting them back to Cartesian coordinates after normalization.
#     # Together with four corner pixels of the other image, we can get the size of new image plane.
#     # Then, remap both image to new image plane and blend two images using Alpha Blending.
#     # TODO
#     height, width, channels = img_1.shape
#     print(f'width={width},height={height}')
#     corners = np.array([[0,0,1],[0,height,1],[width,0,1],[width,height,1]])
#     print(corners)
#     transformed_corners = np.dot(best_homo,corners.T).T
#     print(transformed_corners)
#     transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2].reshape(-1, 1)
#     print(transformed_corners)
#     # Find the size of the new image plane
#     min_x = int(np.floor(min(transformed_corners[:, 0])))
#     max_x = int(np.ceil(max(transformed_corners[:, 0])))
#     min_y = int(np.floor(min(transformed_corners[:, 1])))
#     max_y = int(np.ceil(max(transformed_corners[:, 1])))
#     min_x = min(min_x,0)
#     max_x = max(max_x,height)
#     min_y = min(min_y,0)
#     max_y = max(max_y,width)


#     # Calculate the size of the new image
#     new_height = max_x - min_x
#     new_width = max_y - min_y
#     print(f'new_width={new_width},new_height={new_height}')
#     # Create an empty canvas for the new image
#     new_img = np.zeros((new_width, new_height, channels), dtype=np.uint8)
    
#     # Define the translation matrix to move the transformed image within the new image canvas
#     translation_matrix = np.array([[1, 0, -min_x],
#                                    [0, 1, -min_y],
#                                    [0, 0, 1]],dtype=np.float32)

#     # Warp img_1 onto the new image plane
#     warped_img_1 = cv2.warpPerspective(img_1, np.dot(translation_matrix,best_homo), (new_width, new_height))
    
#     inverse_translation_matrix = np.array([[1, 0, -min_y],
#                                    [0, 1, -min_x],
#                                    [0, 0, 1]],dtype=np.float32)
#     # Warp img_2 onto the new image plane
#     warped_img_2 = cv2.warpPerspective(img_2, inverse_translation_matrix, (new_width, new_height))
    
#     # Perform alpha blending
#     alpha = 0.5  # Adjust the blending factor as needed
#     blended_image = cv2.addWeighted(warped_img_1, alpha, warped_img_2, 1-alpha, 0)
    
#     return blended_image

def project_corners(img_shape, est_homo):
    # Create a matrix of the four corner pixels of the first image in homogeneous coordinates
    corners = np.array([[0, 0, 1], [0, img_shape[1]-1, 1], [img_shape[0]-1, 0, 1], [img_shape[0]-1, img_shape[1]-1, 1]])

    projected_corners = np.dot(corners, est_homo.T)

    return projected_corners

def stitch_blend(img_1,img_2,est_homo):
    corners_1 = project_corners(img_1.shape[:2], est_homo)

    # turn corners_1 to a Cartesian coordinate
    corner_1 = np.array([[cor[0]/cor[2], cor[1]/cor[2]] for cor in corners_1])

    corners_2 = np.array([[0, 0], [0, img_2.shape[1]], [img_2.shape[0], 0], [img_2.shape[0], img_2.shape[1]]])


    # get the boundary of new figure
    min_y = int(min(np.min(corner_1[:, 1]), np.min(corners_2[:, 1])))
    max_y = int(max(np.max(corner_1[:, 1]), np.max(corners_2[:, 1])))
    max_x = int(max(np.max(corner_1[:, 0]), np.max(corners_2[:, 0])))
    min_x = int(min(np.min(corner_1[:, 0]), np.min(corners_2[:, 0])))

    new_x, new_y = max_x, max_y
    offset_x = offset_y = 0

    if min_x < 0:
        offset_x = np.abs(int(min_x))
        new_x = max_x + offset_x

    if min_y < 0:
        offset_y = np.abs(int(min_y))
        new_y = max_y + offset_y
 
    new_img = np.zeros((new_x, new_y, 3))
    # plot the img2, which shouldn't be modified:
    new_img[offset_x:offset_x+img_2.shape[0],offset_y:offset_y+img_2.shape[1]] = img_2

    h, w, _ = new_img.shape
    coordinates = np.array([[x-offset_x,y-offset_y,1] for x in range(h) for y in range(w)])
    inv = np.linalg.inv(est_homo)
    homo_idx_img1 = np.dot(inv,coordinates.T).T
    homo_idx_img1 = homo_idx_img1[:, :2] / homo_idx_img1[:, 2][:, np.newaxis]

    alpha = 0.5
    # plot the img1, which should be affined:
    # the for-loop is extremely slow, but whatever
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            for t in range(3):
                value = interplation(homo_idx_img1[i*w+j], img_1, t)    
                if new_img[i][j][t] != 0 and value!=0: 
                    new_img[i][j][t] = alpha * value + (1-alpha) * new_img[i][j][t]
                # draw the img1
                elif value == 0:
                    continue
                else:
                    new_img[i][j][t] = value  
                       
    new_img = np.uint8(new_img)
    return new_img


def interplation(idx, img, t):
    x, y = idx
    if x<0 or y<0 or x >= img.shape[0] or y >= img.shape[1]:
        return 0
    x_f = int(np.floor(x))
    x_c = int(np.ceil(x))
    y_f = int(np.floor(y))
    y_c = int(np.ceil(y))
    if x_f<0 or y_f <0 or x_c >= img.shape[0] or y_c >= img.shape[1]:
        return 0
    else:
        value = int(img[x_f][y_c][t]/4 + img[x_f][y_f][t]/4 + img[x_c][y_f][t]/4 + img[x_c][y_c][t]/4)
        return value

def generate_panorama(ordered_img_seq):
    # finally we can use \verb|feature_matching| \verb|align_pair| and \verb|stitch_blend| to generate 
    # panorama with multiple images
    # TODO
    panorama = ordered_img_seq[0]
    for i in range(len(ordered_img_seq)):
        if i==0:
            continue
        else:
            pix_1,pix_2 = feature_matching(cv2.cvtColor(panorama,cv2.COLOR_BGR2GRAY),cv2.cvtColor(ordered_img_seq[i],cv2.COLOR_BGR2GRAY))
            homo_matrix = align_pair(pix_1,pix_2)

            panorama = stitch_blend(panorama,ordered_img_seq[i],homo_matrix)
            print(f'{i}th image has been stitched!')
            # resize the panorama 
            # panorama = cv2.resize(panorama,(800,600))
    return panorama


if __name__ == '__main__':
    pass
    # make image list
    # call generate panorama and it should work well
    # save the generated image following the requirements
    
