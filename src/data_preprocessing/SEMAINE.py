import os

import pandas as pd

from src.data_preprocessing.RECOLA import load_labels, extract_faces_with_filenames_in_dataframe


def main():
    # here we use the functions from the RECOLA.py file, since the preprocessing procedure is the same for SEMAINE as well
    # all we need is to modify paths to data, labels, and so on.
    path_to_data = r"G:\Datasets\SEMAINE\original\SEMAINE_videos"
    output_path = r"G:\Datasets\SEMAINE\preprocessed"
    path_to_arousal_labels = r"G:\Datasets\SEMAINE\SEM_labels_arousal_100Hz_gold_shifted.csv"
    path_to_valence_labels = r"G:\Datasets\SEMAINE\SEM_labels_valence_100Hz_gold_shifted.csv"
    # load arousal and valence labels
    arousal_labels = load_labels(path_to_arousal_labels)
    arousal_labels.columns = ["filename", "timestamp", "arousal"]
    valence_labels = load_labels(path_to_valence_labels)
    valence_labels.columns = ["filename", "timestamp", "valence"]
    # merge arousal and valence labels
    labels = pd.merge(arousal_labels, valence_labels, on=["filename", "timestamp"])
    # change the dtype of timestamp column to float
    labels["timestamp"] = labels["timestamp"].astype(float)
    # extract faces
    metadata = extract_faces_with_filenames_in_dataframe(df_with_filenames=labels, path_to_data=path_to_data,
                                              output_path=output_path, every_n_frames=20)
    # delete nan values from the metadata
    metadata = metadata.dropna()
    # save metadata
    metadata.to_csv(os.path.join(r"G:\Datasets\SEMAINE", "preprocessed_labels.csv"), index=False)



if __name__ == '__main__':
    main()