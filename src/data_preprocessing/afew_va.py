import json
import os
from typing import Dict

import pandas as pd


def get_arousal_and_valence_for_every_frame_in_one_video(json_filename:str)->pd.DataFrame:
    result_df = pd.DataFrame(columns=['frame_num','arousal','valence'])
    with open(json_filename, "r") as f:
        data = json.load(f)
        data = data['frames']
        for key, value in data.items():
            result_df.loc[len(result_df.index)] = [key, value['arousal'], value['valence']]
    return result_df


def get_arousal_and_valence_all_videos(folder_path:str)->Dict[str, pd.DataFrame]:
    result_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            result_dict[filename] = get_arousal_and_valence_for_every_frame_in_one_video(os.path.join(folder_path, filename))
    return result_dict



def main():
    path_dir = r'G:\Datasets\AFEW-VA\AFEW-VA\AFEW-VA\data'
    labels = get_arousal_and_valence_all_videos(path_dir)
    print(labels)

if __name__ == "__main__":
    main()