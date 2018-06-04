import pandas as pd
import numpy as np
import debacl as dcl
import sys

from sklearn.cluster import DBSCAN
from itertools import cycle
from operator import itemgetter

pd.options.mode.chained_assignment = None

def standardize(df):
    """
    Inputs: Dataframe
    Outputs: Dataframe, list of tuples storing the mean and standard deviation for each column

    This function returns a dataframe with standardized initial_speed, break_x_full, and break_z_full columns as well
    as the stats needed to calculate original values.
    """
    col_stats = []
    for col in ['initial_speed', 'break_x_full', 'break_z_full']:  # These are the pitch attributes used for clustering
        col_mean_std = (df[col].mean(), df[col].std())
        col_stats.append(col_mean_std)
        df[col] = (df[col] - col_mean_std[0]) / col_mean_std[1]
    return df, col_stats


def calculate_centroid(df, label):
    """
    Inputs: Dataframe, pitch label
    Outputs: A two-item list with the pitch label and the coordinates of that label's centroid

    This function calculates the coordinates of the centroid for a set of pitches with the same label.
    """
    newdf = df[df['labels'] == label]
    centroid = [label, (newdf['initial_speed'].mean(), newdf['break_x_full'].mean(), newdf['break_z_full'].mean())]
    return centroid


def find_all_centroids(df):
    """
    Inputs: Dataframe
    Outputs: A list of lists where each item is the pitch label and that label's centroid

    This function returns a list of lists, each list containing a different pitch label the coordinates of the centroid.
    """
    all_centroids = []
    for label in df['labels'].unique():
        if label > -0.5:
            # If a point is not assigned a cluster, it gets a label of -1.
            # We do not want to include those in this function.
            all_centroids.append(calculate_centroid(df, label))
    return all_centroids


def closest_centroid(row, all_centroids):
    """
    Inputs: Row of a dataframe, list of all_centroids
    Outputs: Row of a dataframe

    This function takes in a row of the pitchers dataframe, the output of the find_all_centroids function.
    For any pitch that was misclassified by the algorithm (given a value of -1), this function re-classifies it
    to the label of the nearest centroid, using Euclidean distance.
    """
    if row['labels'] == -1:
        point = (row['initial_speed'], row['break_x_full'], row['break_z_full'])
        closest_label = -1
        distance = 100000
        for i in range(len(all_centroids)):
            new_dist = np.sqrt(((point[0] - all_centroids[i][1][0]) ** 2) +
                               ((point[1] - all_centroids[i][1][1]) ** 2) +
                               ((point[2] - all_centroids[i][1][2]) ** 2))
            if new_dist < distance:
                distance = new_dist
                closest_label = all_centroids[i][0]
        row['labels'] = closest_label
    return row


def label_fixer(df):
    """
    Inputs: Dataframe
    Outputs: Dataframe

    This function takes in a dataframe, finds all centroids for each pitcher, and runs the closest_centroid function
    on all rows
    """
    all_centroids = find_all_centroids(df)
    clean_df = df.apply(lambda row: closest_centroid(row, all_centroids), axis=1)
    return clean_df


def label_creator(pitcher_matrix, k, prune_threshold):
    """
    Inputs: Matrix, k (an integer), prune_threshold (an integer)
    Outputs: List of labels where the length of the list is equal to the number of rows in the matrix

    This function takes in a matrix and two level-set tree parameters.  K is the number of nearest neighbors used to
    build the k-nearest neighbors graph used in the level-set trees algorithm.  Prune_threshold is the threshold for
    minimum nodes in a leaf.  If there are fewer nodes than the prune threshold, the leaf is merged with its siblings.
    This function runs the level-set tree algorithm and returns a list of labels, indexed to match the input matrix.
    """
    tree = dcl.construct_tree(pitcher_matrix, k, prune_threshold)
    labels = tree.get_clusters(method='leaf', fill_background=True)
    return labels[:, 1]


def df_to_matrix(df):
    """
    Inputs: Dataframe
    Outputs: Matrix

    This functions takes in a dataframe, selects only the 'intial_speed', 'break_x_full', and 'break_z_full' columns,
    and returns those columns as a matrix to be input to the level-set trees algorithm.
    """
    attrs = df[['initial_speed', 'break_x_full', 'break_z_full']]
    pitcher_matrix = pd.DataFrame.as_matrix(attrs)
    return pitcher_matrix


def classify_pitches(df, pitcher, k, prune_threshold):
    """
    Inputs: Dataframe, the name of a pitcher, k, and prune_threshold
    Outputs: Dataframe with labels

    This function uses the helper functions above to slice the dataframe pitcher-by-pitcher and predict labels for each
    row.
    """
    pitcher_df = df[df['pitcher_name'] == pitcher]
    pitcher_df, _ = standardize(pitcher_df)
    pitcher_matrix = df_to_matrix(pitcher_df)
    pitcher_df['labels'] = label_creator(pitcher_matrix, k, prune_threshold)
    return pitcher_df


def distance_predictor(df):
    """
    Inputs: Dataframe
    Outputs: List of lists where each item contains a pair of centroids and the distance between them

    This function takes a dataframe and calculates the distance between each pair of centroids.
    """
    all_centroids = find_all_centroids(df)
    predicted_distances = []
    for i in range(len(all_centroids)):
        for j in range(i + 1, len(all_centroids)):
            distance = np.sqrt(((all_centroids[i][1][0] - all_centroids[j][1][0]) ** 2) +
                               ((all_centroids[i][1][1] - all_centroids[j][1][1]) ** 2) +
                               ((all_centroids[i][1][2] - all_centroids[j][1][2]) ** 2))
            distance_info = [all_centroids[i][0], all_centroids[j][0], distance]
            predicted_distances.append(distance_info)
    return predicted_distances


def pitch_joiner(df, distance_list, distance_threshold):
    """
    Inputs: Dataframe, list os lists (output of distance_predictor), distance threshold
    Outputs: Dataframe

    If two cluster centroids are closer together than the distance_threshold, this function gives merges them and gives
    them the same label.
    """
    sorted_distance_list = sorted(distance_list, key=itemgetter(2))
    if sorted_distance_list[0][2] < distance_threshold:
        df.loc[df['labels'] == sorted_distance_list[0][1], 'labels'] = sorted_distance_list[0][0]
    return df


def pitch_merger(df, distance_threshold):
    """
    Inputs: Dataframe, distance threshold
    Outputs: Dataframe

    This is a recursive function that performs the pitch_joiner function until the number of unique labels no longer
    changes.
    """
    num_clusters = len(df['labels'].unique())
    distance_list = distance_predictor(df)
    df = pitch_joiner(df, distance_list, distance_threshold)
    if len(df['labels'].unique()) != num_clusters:
        df = pitch_merger(df, distance_threshold)
        return df
    else:
        return df


def outlier_finder(df, eps, min_samples):
    """
    Inputs: Dataframe, epsilon, min_samples
    Outputs: Dataframe

    This function checks to see if there was a clear cluster that was mislabeled as part of another cluster.  It uses
    DBSCAN (density-based clustering) with two inputs: epsilon (how close two points need to be to be conisdered a
    cluster) and min_samples (the minimum number of points required to be a cluster).  If there is a mislabeled cluster,
    it gets relabeled.
    """
    pitch_labels = df['labels'].unique()
    df_list = []
    for pitch in pitch_labels:
        open_labels = free_labels(pitch_labels)
        pitch_df = df[df['labels'] == pitch]
        pitch_matrix = df_to_matrix(pitch_df)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pitch_matrix)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            new_labels = unique_label_maker(labels, open_labels)
            pitch_df['labels'] = new_labels
        pitch_df = label_fixer(pitch_df)
        df_list.append(pitch_df)
    output_df = pd.concat(df_list)
    return output_df


def free_labels(label_list):
    """
    Inputs: List of used labels
    Outputs: List of unused labels

    This function finds the labels that have not been assigned to any points.
    """
    all_labels = list(range(1000))
    used_labels = label_list
    unused_labels = list(set(all_labels) - set(used_labels))
    return unused_labels


def unique_label_maker(proposed_labels, unused_labels):
    """
    Inputs: List of potential labels, list of free labels
    Outputs: List of labels

    This function checks whether or not a proposed label is already in use.  If so, it will use the next free label
    instead.
    """
    proposals = list(set(proposed_labels))
    for i in range(len(proposed_labels)):
        if proposed_labels[i] != -1:
            label_index = proposals.index(proposed_labels[i])
            proposed_labels[i] = unused_labels[label_index]
    return proposed_labels


def cluster(df, k, prune_threshold, distance_threshold, eps, min_samples):
    """
    Inputs: Dataframe, k, prune_threshold, distance_threshold, epsilon, min_samples
    Outputs: Dataframe

    This function uses all of the helper functions above to start with a raw dataframe and output a dataframe with a
    new column which are the predicted labels.
    """
    pitcher_names = df['pitcher_name'].unique()
    df_list = []
    counter = 1
    for pitcher in pitcher_names:
        print("Clustering pitcher " + str(counter) + " of " + str(len(pitcher_names)))
        pitcher_df = classify_pitches(df, pitcher, k, prune_threshold)
        pitcher_df['reclass'] = np.where(pitcher_df['labels'] == -1, 1, 0)
        if len(pitcher_df['labels'].unique()) > 1:
            pitcher_df = pitch_merger(pitcher_df, distance_threshold)
        pitcher_df = label_fixer(pitcher_df)
        pitcher_df = outlier_finder(pitcher_df, eps, min_samples)
        df_list.append(pitcher_df)
        counter += 1
    output_df = pd.concat(df_list)
    return output_df


def df_to_csv(df):
    """
    Inputs: Dataframe
    Outputs: CSV

    This function takes a dataframe and saves a CSV called "pitch_clusters.csv" in the working directory
    """
    df.to_csv('pitch_clusters.csv')

if __name__ == "__main__":
    pitchers = pd.read_csv(sys.argv[1])
    cluster_df = cluster(df=pitchers,
                     k=23,
                     prune_threshold=5,
                     distance_threshold=0.8,
                     eps=0.8,
                     min_samples=5)
    df_to_csv(cluster_df)