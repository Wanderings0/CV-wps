from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
def visualize_matches(I1, I2, matches):
    # display two images side-by-side with matches
    # this code is to help you visualize the matches, you don't need
    # to use it to produce the results for the assignment

    I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
    I3[:,:I1.size[0],:] = I1
    I3[:,I1.size[0]:,:] = I2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3).astype(np.uint8))
    ax.plot(matches[:,0],matches[:,1],  '+r')
    ax.plot( matches[:,2] + I1.size[0], matches[:,3], '+r')
    ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
    plt.show()


def normalize_points(pts):
    # Normalize points of eight point algorithm
    # 1. calculate mean and std
    # 2. build a transformation matrix
    # :return normalized_pts: normalized points
    # :return T: transformation matrix from original to normalized points
    # 1. calculate mean and std
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)

    # 2. build a transformation matrix
    T = np.array([[1/std[0], 0, -mean[0]/std[0]],
                  [0, 1/std[1], -mean[1]/std[1]],
                  [0, 0, 1]])

    # Apply the transformation to normalize the points
    normalized_pts = np.matmul(T, np.column_stack((pts, np.ones((pts.shape[0], 1)))).T).T
    # print(normalized_pts)
    return normalized_pts, T

def fit_fundamental(matches):
    # Calculate fundamental matrix from ground truth matches
    # 1. (normalize points if necessary)
    # 2. (x2, y2, 1) * F * (x1, y1, 1)^T = 0 -> AX = 0
    # X = (f_11, f_12, ..., f_33) 
    # build A(N x 9) from matches(N x 4) according to Eight-Point Algorithm
    # 3. use SVD (np.linalg.svd) to decomposite the matrix
    # 4. take the smallest eigen vector(9, ) as F(3 x 3)
    # 5. use SVD to decomposite F, set the smallest eigenvalue as 0, and recalculate F
    # 6. Report your fundamental matrix results

    pts1 = matches[:, :2]
    pts2 = matches[:, 2:]

    normalized_pts1, T1 = normalize_points(pts1)
    normalized_pts2, T2 = normalize_points(pts2)
    # if without normalization
    # normalized_pts1 = np.column_stack((pts1, np.ones((pts1.shape[0], 1))))
    # normalized_pts2 = np.column_stack((pts2, np.ones((pts2.shape[0], 1))))
    N = len(matches)
    A = np.zeros((N, 9))
    for i in range(N):
        A[i, :] = np.kron(normalized_pts2[i], normalized_pts1[i])

    # 3. use SVD (np.linalg.svd) to decompose the matrix
    _, _, V = np.linalg.svd(A)

    # 4. take the smallest eigen vector(9, ) as F(3 x 3)
    F = V[-1].reshape(3, 3)

    # 5. use SVD to decompose F, set the smallest eigenvalue as 0, and recalculate F
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    # Denormalize F
    F = np.dot(T2.T, np.dot(F, T1))

    return F

def visualize_fundamental(matches, F, I1, I2):
    # Visualize the fundamental matrix in image 2
    N = len(matches)
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1, np.kron(np.ones((3,1)), l).transpose())   # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis = 1)
    closest_pt = matches[:, 2:4] - np.multiply(L[:, 0:2],np.kron(np.ones((2, 1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]] * 10    # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I2).astype(np.uint8))
    ax.plot(matches[:, 2],matches[:, 3],  '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]],[matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]],[pt1[:, 1], pt2[:, 1]], 'g')
    ax.axis('off')
    plt.show()
    

def evaluate_fundamental(matches, F):
    N = len(matches)
    points1, points2 = matches[:, :2], matches[:, 2:]
    points1_homogeneous = np.concatenate([points1, np.ones((N, 1))], axis=1)
    points2_homogeneous = np.concatenate([points2, np.ones((N, 1))], axis=1)
    product = np.dot(np.dot(points2_homogeneous, F), points1_homogeneous.T)
    diag = np.diag(product)
    residual = np.mean(diag ** 2)
    return residual

## Task 0: Load data and visualize
## load images and match files for the first example
## matches[:, :2] is a point in the first image
## matches[:, 2:] is a corresponding point in the second image

library_image1 = Image.open('data/library1.jpg')
library_image2 = Image.open('data/library2.jpg')
library_matches = np.loadtxt('data/library_matches.txt')

lab_image1 = Image.open('data/lab1.jpg')
lab_image2 = Image.open('data/lab2.jpg')
lab_matches = np.loadtxt('data/lab_matches.txt')

## Visualize matches
# visualize_matches(library_image1, library_image2, library_matches)
# visualize_matches(lab_image1, lab_image2, lab_matches)

## Task 1: Fundamental matrix
## display second image with epipolar lines reprojected from the first image

# first, fit fundamental matrix to the matches
# Report your fundamental matrices, visualization and evaluation results
library_F = fit_fundamental(library_matches) # this is a function that you should write
# visualize_fundamental(library_matches, library_F, library_image1, library_image2)
# print(evaluate_fundamental(library_matches, library_F))
assert evaluate_fundamental(library_matches, library_F) < 0.5

lab_F = fit_fundamental(lab_matches) # this is a function that you should write
# visualize_fundamental(lab_matches, lab_F, lab_image1, lab_image2) 
# print(evaluate_fundamental(lab_matches, lab_F))
assert evaluate_fundamental(lab_matches, lab_F) < 0.5

print('Task 1 is completed!')

# exit()
## Task 2: Camera Calibration

def calc_projection(points_2d, points_3d):
    # Calculate camera projection matrices
    # 1. Points_2d = P * Points_3d -> AX = 0
    # X = (p_11, p_12, ..., p_34) is flatten of P
    # build matrix A(2*N, 12) from points_2d
    # 2. SVD decomposite A
    # 3. take the eigen vector(12, ) of smallest eigen value
    # 4. return projection matrix(3, 4)
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return P: projection matrix
    # Construct matrix A
    N = points_2d.shape[0]
    A = np.zeros((2*N, 12))
    for i in range(N):
        x, y, z = points_3d[i]
        u, v = points_2d[i]
        A[2*i] = [-x, -y, -z, -1, 0, 0, 0, 0, u*x, u*y, u*z, u]
        A[2*i+1] = [0, 0, 0, 0, -x, -y, -z, -1, v*x, v*y, v*z, v]
    # SVD decomposition
    U, S, V = np.linalg.svd(A)
    P = V[-1].reshape(3, 4)
    return P

def rq_decomposition(P):
    # Use RQ decomposition to calculte K, R, T
    # 1. perform QR decomposition on left-most 3x3 matrix of P(3 x 4) to get K, R
    # 2. normalize to set K[2, 2] = 1
    # 3. calculate T by P = K[R|T]
    # :param P: projection matrix
    # :return K, R, T: camera matrices
    K, R = np.linalg.qr(P[:, :3])
    K = K / K[2, 2]
    T = np.linalg.inv(K)@P[:, 3]

    return K, R, T

def evaluate_points(P, points_2d, points_3d):
    # Visualize the actual 2D points and the projected 2D points calculated from
    # the projection matrix
    # You do not need to modify anything in this function, although you can if you
    # want to
    # :param P: projection matrix 3 x 4
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return points_3d_proj: project 3D points to 2D by P
    # :return residual: residual of points_3d_proj and points_2d

    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(P, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

def triangulate_points(P1, P2, point1, point2):
    # Use linear least squares to triangulation 3d points
    # 1. Solve: point1 = P1 * point_3d
    #           point2 = P2 * point_3d
    # 2. use SVD decomposition to solve linear equations
    # :param P1, P2 (3 x 4): projection matrix of two cameras
    # :param point1, point2: points in two images
    # :return point_3d: 3D points calculated by triangulation

    A = np.vstack((point1[1]*P1[2, :] - P1[1, :],
                P1[0, :] - point1[0]*P1[2, :],
                point2[1]*P2[2, :] - P2[1, :],
                P2[0, :] - point2[0]*P2[2, :]))

    # Use SVD decomposition to solve linear equations
    _, _, V = np.linalg.svd(A)

    # Extract the 3D coordinates from the last column of V
    point_3d = V[-1, :3] / V[-1, 3]

    return point_3d


lab_points_3d = np.loadtxt('data/lab_3d.txt')

projection_matrix = dict()
for key, points_2d in zip(["lab_a", "lab_b"], [lab_matches[:, :2], lab_matches[:, 2:]]):
    P = calc_projection(points_2d, lab_points_3d)
    points_3d_proj, residual = evaluate_points(P, points_2d, lab_points_3d)
    distance = np.mean(np.linalg.norm(points_2d - points_3d_proj))
    # print("Residual of {} is {}".format(key, residual))
    # print("Distance of {} is {}".format(key, distance))
    # Check: residual should be < 20 and distance should be < 4 
    assert residual < 20.0 and distance < 4.0
    projection_matrix[key] = P
print("Task 2 is completed!")
# exit()

## Task 3
## Camera Centers
projection_library_a = np.loadtxt('data/library1_camera.txt')
projection_library_b = np.loadtxt('data/library2_camera.txt')
projection_matrix["library_a"] = projection_library_a
projection_matrix["library_b"] = projection_library_b
print(projection_matrix.keys())
for P in projection_matrix.values():
    
    # Paste your K, R, T results in your report
    K, R, T = rq_decomposition(P)

    # print("K: \n", K)
    # print("R: \n", R)
    # print("T: \n", T)
    # print("sep")

    # Check: K should be a valid intrinsics matrix
    # assert np.allclose(np.dot(K, K.T), np.dot(K.T, K))
    # # assert np.all(np.linalg.eigvals(K) > 0)

    # # Check: T should have positive z
    # assert T[2] > 0

    # # Check: P should be equal to K[R|T]
    # assert np.allclose(P, np.dot(K, np.hstack((R, T.reshape(3, 1)))))
print("Task 3 is completed!")

## Task 4: Triangulation
lab_points_3d_estimated = []
for point_2d_a, point_2d_b, point_3d_gt in zip(lab_matches[:, :2], lab_matches[:, 2:], lab_points_3d):
    point_3d_estimated = triangulate_points(projection_matrix['lab_a'], projection_matrix['lab_b'], point_2d_a, point_2d_b)

    # Residual between ground truth and estimated 3D points
    residual_3d = np.sum(np.linalg.norm(point_3d_gt - point_3d_estimated))
    assert residual_3d < 0.1
    lab_points_3d_estimated.append(point_3d_estimated)

# Residual between re-projected and observed 2D points
lab_points_3d_estimated = np.stack(lab_points_3d_estimated)
_, residual_a = evaluate_points(projection_matrix['lab_a'], lab_matches[:, :2], lab_points_3d_estimated)
_, residual_b = evaluate_points(projection_matrix['lab_b'], lab_matches[:, 2:], lab_points_3d_estimated)
assert residual_a < 20 and residual_b < 20
print('Residual of lab_a: ', residual_a)
print('Residual of lab_b: ', residual_b)

library_points_3d_estimated = []
for point_2d_a, point_2d_b in zip(library_matches[:, :2], library_matches[:, 2:]):
    point_3d_estimated = triangulate_points(projection_matrix['library_a'], projection_matrix['library_b'], point_2d_a, point_2d_b)
    library_points_3d_estimated.append(point_3d_estimated)

# Residual between re-projected and observed 2D points
library_points_3d_estimated = np.stack(library_points_3d_estimated)
_, residual_a = evaluate_points(projection_matrix['library_a'], library_matches[:, :2], library_points_3d_estimated)
_, residual_b = evaluate_points(projection_matrix['library_b'], library_matches[:, 2:], library_points_3d_estimated)
assert residual_a < 30 and residual_b < 30
print('Residual of library_a: ', residual_a)
print('Residual of library_b: ', residual_b)

print("Task 4 is completed!")

## Task 5: Fundamental matrix estimation without ground-truth matches
import cv2

def fit_fundamental_without_gt(image1, image2):
    # Calculate fundamental matrix without groundtruth matches
    # 1. convert the images to gray
    # 2. compute SIFT keypoints and descriptors
    # 3. match descriptors with Brute Force Matcher
    # 4. select good matches
    # 5. extract matched keypoints
    # 6. compute fundamental matrix with RANSAC
    # :param image1, image2: two-view images
    # :return fundamental_matrix
    # :return matches: selected matched keypoints 
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match descriptors with Brute Force Matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute fundamental matrix with RANSAC
    fundamental_matrix, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

    # Return fundamental matrix and matches
    return fundamental_matrix, np.concatenate((src_pts.squeeze(), dst_pts.squeeze()), axis=1)[mask.ravel() == 1]



house_image1 = Image.open('data/house1.jpg')
house_image2 = Image.open('data/house2.jpg')

house_F, house_matches = fit_fundamental_without_gt(np.array(house_image1), np.array(house_image2))
# print('house_F: \n', house_F)
# print('house_matches: \n', house_matches)
# print(len(house_matches))
visualize_fundamental(house_matches, house_F, house_image1, house_image2)
#  compute the average residual for the inliers
residual = evaluate_fundamental(house_matches, house_F)
print('Residual of house: ', residual)
# the number of inliers
print('Number of inliers: ', len(house_matches))
# visualize the matches
visualize_matches(house_image1, house_image2, house_matches)



gaudi_image1 = Image.open('data/gaudi1.jpg')
gaudi_image2 = Image.open('data/gaudi2.jpg')

gaudi_F, gaudi_matches = fit_fundamental_without_gt(np.array(gaudi_image1), np.array(gaudi_image2))
# print('gaudi_F: \n', gaudi_F)
# print('gaudi_matches: \n', gaudi_matches)
# print(len(gaudi_matches))
visualize_fundamental(gaudi_matches, gaudi_F, gaudi_image1, gaudi_image2)
#  compute the average residual for the inliers
residual = evaluate_fundamental(gaudi_matches, gaudi_F)
print('Residual of gaudi: ', residual)
# the number of inliers
print('Number of inliers: ', len(gaudi_matches))
# visualize the matches
visualize_matches(gaudi_image1, gaudi_image2, gaudi_matches)
