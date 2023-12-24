import cv2
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import trimesh
import time

def normalize_disparity_map(disparity_map):
    # Normalize disparity map for visualization
    # disparity should be larger than zero
    return np.maximum(disparity_map, 0.0) / disparity_map.max()
    
def visualize_disparity_map(disparity_map, gt_map,path):
    # Normalize disparity map
    plt.axis('off')
    disparity_map = normalize_disparity_map(disparity_map)
    gt_map = normalize_disparity_map(gt_map)
    # Visualize
    # concat_map = np.concatenate([disparity_map], axis=1)
    # plt.imshow(disparity_map, 'gray')
    plt.imsave(path+'.jpg',disparity_map,cmap='gray')
    plt.close()
    plt.imsave(path+'gt.jpg',gt_map,cmap='gray')
    plt.close()
    # plt.show()

def compute_disparity_map_simple(ref_image, sec_image, window_size, disparity_range, matching_function):
    # 1. Simple Stereo System: image planes are parallel to each other
    # 2. For each row, scan all pixels in that row
    # 3. Generate a window for each pixel
    # 4. Search a disparity(d) in (min_disparity, max_disparity)
    # 5. Select the best disparity that minimize window difference between (row, col) and (row, col - d)
    # 6. implement all the matching function including: ['SSD', 'SAD', 'normalized_correlation']. 
    # That is sum of squared differences (SSD), sum of absolute differences (SAD), and normalized correlation.
    # Initialize the disparity map to zeros
    disparity_map = np.zeros(ref_image.shape, dtype=np.float32)
    
    # Get the image dimensions
    height, width = ref_image.shape
    
    # Set the range of disparities to search
    min_disp, max_disp = disparity_range
    
    half_window = window_size // 2
    
    # Iterate through each pixel in the reference image
    for row in range(half_window, height - half_window):
        for col in range(half_window, width - half_window):
            # Initialize the best disparity for the current pixel
            best_disparity = 0
            if matching_function == 'normalized_correlation':
                max_score = -np.inf  # For normalized correlation, higher is better
            else:
                min_diff = np.inf  # For SSD and SAD, lower is better

            # Generate reference window
            ref_window = ref_image[row - half_window:row + half_window + 1,
                                   col - half_window:col + half_window + 1]

            # Search within the disparity range
            for d in range(min_disp, max_disp):
                # Make sure we do not go out of bounds
                if col - d < half_window or col - d >= width - half_window:
                    continue

                # Generate secondary window
                sec_window = sec_image[row - half_window:row + half_window + 1,
                                       col - d - half_window:col - d + half_window + 1]

                # Calculate window difference
                if matching_function == 'SSD':
                    diff = np.sum((ref_window - sec_window) ** 2)
                elif matching_function == 'SAD':
                    diff = np.sum(np.abs(ref_window - sec_window))
                elif matching_function == 'normalized_correlation':
                    # Normalized correlation formula
                    
                    norm_ref_window = ref_window - np.mean(ref_window)
                    ref_std = np.std(norm_ref_window)
                    norm_sec_window = sec_window - np.mean(sec_window)
                    sec_std = np.std(norm_sec_window)
                    score = np.sum(norm_ref_window * norm_sec_window)
                    score /= ref_std*sec_std
                    
                    # Check if this score is better
                    if score > max_score:
                        max_score = score
                        best_disparity = d
                    continue

                # Check if this disparity is better
                if diff < min_diff:
                    min_diff = diff
                    best_disparity = d

            # Assign the best disparity to the disparity map
            disparity_map[row, col] = best_disparity

    return disparity_map

def simple_disparity(ref_image, second_image, gt_map,ori_img):
    # 1. Change window size, disparity range and matching functions
    # 2. Report the disparity maps and running time
    
    window_sizes = [13]  # Try different window sizes
    disparity_range = (0, 16)  # Determine appropriate disparity range
    matching_functions = ['SSD', 'SAD', 'normalized_correlation']  # Try different matching functions
    
    # Generate disparity maps for different settings
    for window_size in window_sizes:
        for matching_function in matching_functions:
            start_time = time.time() 
            disparity_map = compute_disparity_map_simple(ref_image, second_image, window_size, disparity_range, matching_function)
            end_time = time.time()
            running_time = end_time - start_time 
            print(f"Window Size: {window_size}, Matching Function: {matching_function}, Running Time: {running_time:.2f} seconds",flush=True)
            path = 'dp/tsukuba_window['+str(window_size)+']'+'fun['+matching_function+']'
            visualize_disparity_map(disparity_map,gt_map,path=path)
            visualize_pointcloud(ori_img,disparity_map,path)



def compute_depth_map(disparity_map, baseline, focal_length):
    # Create a mask for pixels where the disparity is zero or too low
    valid_disparity_mask = disparity_map > 1
    
    # Initialize depth map with infinities or a large value
    depth_map = np.ones(disparity_map.shape)*100 # Large number to represent "infinity"
    
    # Compute depth only for non-zero disparities
    depth_map[valid_disparity_mask] = focal_length * baseline / disparity_map[valid_disparity_mask]
    return depth_map

def compute_depth(disparity, baseline, focal_length):
    if disparity == 0:
        return 0
    else:
        return baseline * focal_length / disparity

def visualize_pointcloud(ref_image, disparity_map,path):
    baseline = 100
    focal_length = 10  

    # Calculate depth map from disparity
    depth_map = compute_depth_map(disparity_map, baseline, focal_length)

    height, width = ref_image.shape[:2]
    points = []
    colors = []
    for y in range(height):
        for x in range(width):
            # disparity = disparity_map[y, x]
            # depth = compute_depth(disparity,baseline,focal_length)
            depth = depth_map[y,x]
            if depth > 0:
                color = ref_image[y, x]
                # print(color)
                points.append([width-x, height-y, depth])
                colors.append(color)
                # colors.append([color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0])


    # Convert to numpy arrays for trimesh
    points = np.array(points)
    colors = np.array(colors)

    # Save point cloud as ply file using trimesh
    pointcloud = trimesh.points.PointCloud(vertices=points, colors=colors)
    # if path is not None:
    pointcloud.export(path+'_pointcloud.ply')
        # pointcloud.export('pointcloud2.ply')

def sumsqr_diff(block1, block2):
    return np.sum((block1 - block2) ** 2)


def compute_disparity_map_dp(ref_image, sec_image):
    # Dynamic programming stereo matching
    start_time = time.time()
    m, n = ref_image.shape
    disparity_map = np.zeros((m, n), dtype=np.float32)
    win_size = 5
    for k in range(win_size//2, m-win_size//2):
        disparity_space_image = np.zeros((n, n), dtype=np.float32)
        # calculate Match score for patch centered at left pixel i with patch centered at right pixel j
        for j in range(win_size//2, n-win_size//2):
            for i in range(j, min(n-win_size//2, j+64)):
            #for i in range(win_size//2, min(n-win_size//2, j+64)):
                disparity_space_image[j,i] = sumsqr_diff(ref_image[max(0, k-win_size//2):min(m, k+win_size//2+1), max([0, i-win_size//2]):min([i+win_size//2+1, n])], 
                                                                   sec_image[max(0, k-win_size//2):min(m, k+win_size//2+1), max([0, j-win_size//2]):min([j+win_size//2+1, n])], 
                                                                   )

        cost = np.zeros((n, n), dtype=np.float32)
        backtrace = np.zeros((n, n), dtype=np.float32)

        ### 7500 or 8000 is a good choice
        occlusion_cost = 8000

        # calculate cost matrix
        for j in range(win_size//2, n-win_size//2):
            for i in range(j, min(n-win_size//2, j+64)):
                if i == win_size//2 and j == win_size//2:
                    cost[j,i] = disparity_space_image[j,i]
                elif j == win_size//2:
                    cost[j,i] = cost[j,i-1] + occlusion_cost
                    backtrace[j,i] = 3
                elif i == j:
                    min1 = cost[j-1,i-1] + disparity_space_image[j,i]
                    min2 = cost[j-1,i] + occlusion_cost
                    cost[j,i] = min(min1, min2)
                    if min1 == cost[j,i]:
                        backtrace[j,i] = 1
                    else:
                        backtrace[j,i] = 2
                elif i == j + 64:
                    min1 = cost[j-1,i-1] + disparity_space_image[j,i]
                    min3 = cost[j,i-1] + occlusion_cost
                    cost[j,i] = min(min1, min3)
                    if min1 == cost[j,i]:
                        backtrace[j,i] = 1
                    else:
                        backtrace[j,i] = 3
                elif i == n-1-win_size//2:
                    min1 = cost[j-1,i-1] + disparity_space_image[j,i]
                    min2 = cost[j-1,i] + occlusion_cost
                    cost[j,i] = min(min1, min2)
                    if min1 == cost[j,i]:
                        backtrace[j,i] = 1
                    else:
                        backtrace[j,i] = 2
                else:
                    min1 = cost[j-1,i-1]+disparity_space_image[j,i]
                    min2 = cost[j-1,i]+occlusion_cost
                    min3 = cost[j,i-1]+occlusion_cost
                    cost[j,i] = min(min1, min2, min3)
                    if min1 == cost[j,i]:
                        backtrace[j,i] = 1
                    elif min2 == cost[j,i]:
                        backtrace[j,i] = 2
                    else:
                        backtrace[j,i] = 3
        # backtrace
        i = n-1-win_size//2
        j = n-1-win_size//2

        while i >= win_size//2 and j >= win_size//2:
            if backtrace[j,i] == 1:
                disparity_map[k,i] = i-j
                i -= 1
                j -= 1
            elif backtrace[j,i] == 2:
                j -= 1
            else:
                i -= 1
    end_time = time.time()
    print(f'running time is {end_time - start_time}')
    return disparity_map
# Example usage:
# disparity_map = compute_disparity_map_dp(ref_image, sec_image)

if __name__ == "__main__":
    # Read images
    moebius_image1 = cv2.imread("data/moebius1.png")
    moebius_image1_gray = cv2.cvtColor(moebius_image1, cv2.COLOR_BGR2GRAY)
    moebius_image2 = cv2.imread("data/moebius2.png")
    moebius_image2_gray = cv2.cvtColor(moebius_image2, cv2.COLOR_BGR2GRAY)
    moebius_gt = cv2.imread("data/moebius_gt.png", cv2.IMREAD_GRAYSCALE)

    tsukuba_image1 = cv2.imread("data/tsukuba1.jpg")
    tsukuba_image1_gray = cv2.cvtColor(tsukuba_image1, cv2.COLOR_BGR2GRAY)
    tsukuba_image2 = cv2.imread("data/tsukuba2.jpg")
    tsukuba_image2_gray = cv2.cvtColor(tsukuba_image2, cv2.COLOR_BGR2GRAY)
    tsukuba_gt = cv2.imread("data/tsukuba_gt.jpg", cv2.IMREAD_GRAYSCALE)


    # # Task 0: Visualize cv2 Results
    # stereo = cv2.StereoBM.create(numDisparities=64, blockSize=15)
    # moebius_disparity_cv2 = stereo.compute(moebius_image1_gray, moebius_image2_gray)
    # visualize_disparity_map(moebius_disparity_cv2, moebius_gt,path='result/cv2moebius')
    # tsukuba_disparity_cv2 = stereo.compute(tsukuba_image1_gray, tsukuba_image2_gray)
    # visualize_disparity_map(tsukuba_disparity_cv2, tsukuba_gt,path='result/cv2stukuba')


    # # Task 1: Simple Disparity Algorithm
    simple_disparity(tsukuba_image1_gray, tsukuba_image2_gray, tsukuba_gt,tsukuba_image1)
    # # This may run for a long time! 
    # simple_disparity(moebius_image1_gray, moebius_image2_gray, moebius_gt,moebius_image1)
    # print(np.sum(tsukuba_disparity_cv2==0))
    # # Task 2: Compute depth map and visualize 3D pointcloud
    # # You can use the gt/cv2 disparity map first, then try your own disparity map
    # visualize_pointcloud(tsukuba_image1, tsukuba_gt,path = 'result/gtmoebius')
    # visualize_pointcloud(moebius_image1, moebius_gt)

    # # Task 3: Non-local constraints
    # a = compute_disparity_map_dp(tsukuba_image1_gray, tsukuba_image2_gray)
    # visualize_disparity_map(a, tsukuba_gt,path='dp/dp_tsukuba')

