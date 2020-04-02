import numpy as np

def resample_trajectory(tract, new_num_samples):
    dim = tract.shape[1]
    total_dist = 0
    cur_point = tract[0]
    for point in tract[1:]:
        total_dist += np.linalg.norm(point - cur_point)
        cur_point = point

    step_dist = total_dist / (new_num_samples - 1)
    j = 0
    dist_left_on_segment = np.linalg.norm(tract[1] - tract[0])

    new_trajectory = np.zeros((new_num_samples, dim))
    new_trajectory[0] = tract[0]
    for i in range(1, new_num_samples - 1):
        dist_to_walk = step_dist
        while dist_to_walk > 0:
            if dist_left_on_segment < dist_to_walk:
                j += 1
                dist_to_walk -= dist_left_on_segment
                dist_left_on_segment = np.linalg.norm(tract[j + 1] - tract[j])
            else:
                if np.linalg.norm(tract[j + 1] - tract[j]) > 0:
                    new_trajectory[i] = tract[j] + dist_to_walk * (
                            tract[j + 1] - tract[j]) / np.linalg.norm(
                        tract[j + 1] - tract[j])
                else:
                    new_trajectory[i] = tract[j]
                dist_left_on_segment -= dist_to_walk
                break
    new_trajectory[new_num_samples - 1] = tract[len(tract) - 1]
    return new_trajectory

def naive_RMSE(tract, ground_truth):
    dim = len(tract[0])
    if len(tract) > len(ground_truth):
        ground_truth = resample_trajectory(ground_truth, len(tract))
    elif len(ground_truth) > len(tract):
        tract = resample_trajectory(tract, len(ground_truth))

    n = len(tract)
    rmse = 0

    for i in range(n):
        rmse += np.linalg.norm(tract[i] - ground_truth[i]) ** 2

    return np.sqrt(rmse)
