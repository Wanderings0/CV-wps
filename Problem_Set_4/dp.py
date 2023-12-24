import numpy as np
def sumsqr_diff(block1, block2):
    return np.sum((block1.astype(float) - block2.astype(float)) ** 2)

def compute_disparity_map_dp(ref_image, sec_image, disp_range=15, block_size=9, penalty=0.4):

    half_block = block_size // 2
    rows, cols = ref_image.shape
    disparity_map = np.zeros((rows, cols), dtype=np.float32)

    for y in range(half_block, rows - half_block):
        # Initialize the cost and path matrices
        cost = np.ones((cols, disp_range)) * np.inf
        path = np.zeros((cols, disp_range), dtype=np.int32)

        # Build the cost matrix for the current row
        for x in range(half_block, cols - half_block):
            for d in range(disp_range):
                if x - half_block - d >= 0 and x + half_block - d < cols:
                    ref_block = ref_image[y - half_block:y + half_block + 1, x - half_block:x + half_block + 1]
                    sec_block = sec_image[y - half_block:y + half_block + 1, x - half_block - d:x + half_block - d + 1]
                    cost[x, d] = sumsqr_diff(ref_block, sec_block)

        # Apply dynamic programming to find the path of lowest cost
        for x in range(1, cols):
            for d in range(disp_range):
                cost_current = cost[x, d]
                cost_left = cost[x - 1, d]

                # Check the left and diagonal neighbors in the cost matrix
                if d > 0:
                    cost_left_d = cost[x - 1, d - 1] + penalty
                else:
                    cost_left_d = np.inf

                if d < disp_range - 1:
                    cost_left_u = cost[x - 1, d + 1] + penalty
                else:
                    cost_left_u = np.inf

                min_cost = min(cost_left, cost_left_d, cost_left_u)

                # Update cost matrix and path matrix
                cost[x, d] = cost_current + min_cost
                if min_cost == cost_left:
                    path[x, d] = d
                elif min_cost == cost_left_d:
                    path[x, d] = d - 1
                else:
                    path[x, d] = d + 1

        # Backtrack to find the optimal path
        for x in range(cols - half_block - 1, half_block - 1, -1):
            if x == cols - half_block - 1:
                # Start from the lowest cost in the last column
                disparity_map[y, x] = np.argmin(cost[x])
            else:
                # Follow the path matrix
                d = int(disparity_map[y, x + 1])
                disparity_map[y, x] = path[x + 1, d]

    # Scale the disparity values to be in the appropriate range
    disparity_map = (disparity_map - disp_range // 2) * (255 / disp_range)

    return disparity_map

# Example usage:
# disparity_map = compute_disparity_map_dp(ref_image, sec_image)