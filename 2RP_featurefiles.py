import numpy as np
from scipy.spatial.distance import cdist
import random
import os
import copy
import time
import concurrent.futures
import pandas as pd

#Road positive feature points to feature ppoints files


# Calculate the Euclidean distance between two vectors
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


# Calculate the total cost, which is the sum of the minimum distances from all points to the centroids
def total_cost(dataMat, medoids):
    med_idx = medoids["cen_idx"]
    medObject = dataMat[med_idx, :]
    dis = cdist(dataMat, medObject, 'euclidean')
    cost = dis.min(axis=1).sum()
    medoids["t_cost"] = cost


# Assign data points to the nearest centroids, forming clusters
def assign(dataMat, medoids):
    med_idx = medoids["cen_idx"]
    med = dataMat[med_idx]
    dist = cdist(dataMat, med, 'euclidean')
    idx = dist.argmin(axis=1)
    for i in range(len(med_idx)):
        medoids[i] = np.where(idx == i)


# PAM clustering algorithm implementation
def PAM(data, k):
    data = np.mat(data)
    N = len(data)
    cur_medoids = {"cen_idx": random.sample(range(N), k)}
    assign(data, cur_medoids)
    total_cost(data, cur_medoids)
    old_medoids = {"cen_idx": []}
    iter_counter = 1
    while set(old_medoids['cen_idx']) != set(cur_medoids['cen_idx']):
        print("Iteration counter:", iter_counter)
        iter_counter += 1
        best_medoids = copy.deepcopy(cur_medoids)
        old_medoids = copy.deepcopy(cur_medoids)
        for i in range(N):
            for j in range(k):
                if i != j:
                    tmp_medoids = copy.deepcopy(cur_medoids)
                    tmp_medoids["cen_idx"][j] = i
                    assign(data, tmp_medoids)
                    total_cost(data, tmp_medoids)
                    if best_medoids["t_cost"] > tmp_medoids["t_cost"]:
                        best_medoids = copy.deepcopy(tmp_medoids)
        cur_medoids = copy.deepcopy(best_medoids)
        print("Current total cost is:", cur_medoids["t_cost"])
    return cur_medoids


# Load data from a file
def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    return data, len(data)


# Output the centroids to a CSV file
def output_centers(file_path, medoids, data, output_folder):
    cen_idx = medoids['cen_idx']
    output_file_path = os.path.join(output_folder, os.path.basename(file_path).replace('.txt', '.csv'))

    centroids = data[cen_idx, :]

    # Create a DataFrame with column names "Tileid", "ID", "RI", "RJ"
    tileid = [os.path.basename(file_path).split('.')[0]] * len(centroids)  # Use file name as Tileid (without extension)
    ids = list(range(1, len(centroids) + 1))  # Create a list of IDs starting from 1
    df_centroids = pd.DataFrame(centroids, columns=["R_I", "R_J"])
    df_centroids.insert(0, "FpointID", ids)  # Insert ID as the second column
    df_centroids.insert(0, "Tile_id", tileid)  # Insert Tileid as the first column

    # Add the L column, filled with "RN" for all rows
    df_centroids["L"] = ["RP"] * len(centroids)  # All rows filled with "RN"

    # Save the DataFrame as a CSV file
    df_centroids.to_csv(output_file_path, index=False)
    print(f"Centroids saved to {output_file_path}")


# Print the centroid information
def print_medoids(medoids, data):
    cen_idx = medoids['cen_idx']
    centroids = data[cen_idx, :]
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i + 1}: {centroid}")


# Process a single file
def process_file(file_path, output_folder, k):
    start_time = time.time()  # Start timer

    data, num_samples = load_data(file_path)
    print(f"Number of input trajectory points: {num_samples}")
    print(f"Number of clusters chosen: {k}")

    # Check if directory exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # If there are no samples or fewer samples than clusters, handle accordingly
    if num_samples == 0:
        print(f"File {file_path} contains no data, skipping clustering.")
    elif num_samples < k:
        print(
            f"Number of data samples ({num_samples}) is less than the number of clusters ({k}), outputting original data.")
        output_file_path = os.path.join(output_folder, os.path.basename(file_path).replace('.txt', '_centers.csv'))
        np.savetxt(output_file_path, data, fmt='%d', delimiter=',')
    else:
        # Perform PAM clustering
        medoids = PAM(data, k)
        print_medoids(medoids, data)
        output_centers(file_path, medoids, data, output_folder)

    end_time = time.time()  # End timer
    print(f"Processing time for file {file_path}: {end_time - start_time:.2f} seconds")


# Process all files in the folder
def process_files_in_folder(folder_path, output_folder, k_list):
    total_start_time = time.time()  # Start total timer

    for k in k_list:
        print(f"Starting clustering for k = {k}...")
        file_list = os.listdir(folder_path)
        file_paths = [os.path.join(folder_path, file_name) for file_name in file_list if
                      os.path.isfile(os.path.join(folder_path, file_name))]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_file, file_path, output_folder, k) for file_path in file_paths]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    total_end_time = time.time()  # End total timer
    print(f"Total processing time: {total_end_time - total_start_time:.2f} seconds")


# Main program
if __name__ == '__main__':
    input_folder = "./DATA/Rptxt"
    output_folder = "./Featurepointsfiles"

    #RP with SC   the number of road feature points
    k_list = [8]
    process_files_in_folder(input_folder, output_folder, k_list)
