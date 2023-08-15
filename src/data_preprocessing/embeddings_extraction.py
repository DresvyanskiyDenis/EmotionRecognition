from functools import partial
from typing import List, Callable

import torch
from torchvision.io import read_image
import os
import glob
import numpy as np
import pandas as pd
from torchinfo import summary
from tqdm import tqdm

from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor


def construct_and_load_facial_model(path_to_weights:str)->torch.nn.Module:
    model = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8,
                                             num_regression_neurons=2)
    # load weights
    model.load_state_dict(torch.load(path_to_weights))
    # cut off two last layers
    model = torch.nn.Sequential(*list(model.children())[:-2])
    # print the model architecture
    summary(model, (2,3, 224, 224))

    return model


def extract_embeddings_from_images_df(df:pd.DataFrame, model:torch.nn.Module, device:torch.device,
                                      preprocessing_functions:List[Callable],
                                      output_path:str, output_filename:str)->np.ndarray:

    # check if output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # set model to eval mode
    model.eval()
    # move model to device
    model.to(device)
    # create resulting pd.Dataframe
    columns = list(df.columns) + ['emb_{}'.format(i) for i in range(256)]
    embeddings = pd.DataFrame(columns=columns)
    # save empty dataframe to make file visible
    embeddings.to_csv(os.path.join(output_path, '%s.csv'%output_filename), index=False)
    # iterate over images
    for i in tqdm(range(df.shape[0])):
        # get image path
        image_path = df.iloc[i]['path']
        # load image
        image = read_image(image_path)
        # preprocess image
        for preprocessing_function in preprocessing_functions:
            image = preprocessing_function(image)
        # add batch dimension
        image = image.unsqueeze(0)
        # move image to device
        image = image.to(device)
        # extract embeddings
        with torch.no_grad():
            emb = model(image)
            # squeeze embeddings + convert to numpy
            emb = emb.squeeze().cpu().numpy()
        # add embeddings to resulting dataframe. Remember that the embeddings df has 0 rows at the beginning
        new_row = df.iloc[i].values.tolist() + emb.tolist()
        embeddings.loc[i] = new_row
    # save embeddings
    embeddings.to_csv(os.path.join(output_path, '%s.csv'%output_filename), index=False)


def extract_embeddigs_EMOTIC(path_to_data:str, model:torch.nn.Module, preprocessing_functions:List[Callable],
                             output_path:str, output_filename:str, device:torch.device):
    train_labels = pd.read_csv(os.path.join(path_to_data, 'train_labels.csv'))
    dev_labels = pd.read_csv(os.path.join(path_to_data, 'val_labels.csv'))
    test_labels = pd.read_csv(os.path.join(path_to_data, 'test_labels.csv'))
    # change columns name abs_path to path
    train_labels.rename(columns={'abs_path': 'path'}, inplace=True)
    dev_labels.rename(columns={'abs_path': 'path'}, inplace=True)
    test_labels.rename(columns={'abs_path': 'path'}, inplace=True)
    # change the paths to images from '/media/external_hdd_2/Datasets/EMOTIC/' to '/work/home/dsu/Datasets/EMOTIC/'
    train_labels['path'] = train_labels['path'].apply(lambda x: x.replace('/media/external_hdd_2/Datasets/EMOTIC/',
                                                                            '/work/home/dsu/Datasets/EMOTIC/'))
    dev_labels['path'] = dev_labels['path'].apply(lambda x: x.replace('/media/external_hdd_2/Datasets/EMOTIC/',
                                                                        '/work/home/dsu/Datasets/EMOTIC/'))
    test_labels['path'] = test_labels['path'].apply(lambda x: x.replace('/media/external_hdd_2/Datasets/EMOTIC/',
                                                                            '/work/home/dsu/Datasets/EMOTIC/'))
    # extract embeddings
    extract_embeddings_from_images_df(train_labels, model, device, preprocessing_functions, output_path,
                                        output_filename + '_train')
    extract_embeddings_from_images_df(dev_labels, model, device, preprocessing_functions, output_path,
                                        output_filename + '_dev')
    extract_embeddings_from_images_df(test_labels, model, device, preprocessing_functions, output_path,
                                        output_filename + '_test')

def extract_embeddings_SEWA(path_to_data:str, model:torch.nn.Module, preprocessing_functions:List[Callable],
                                output_path:str, output_filename:str, device:torch.device):
    labels = pd.read_csv(os.path.join(path_to_data, 'preprocessed_labels.csv'))
    # change columns name filename to path
    labels.rename(columns={'filename': 'path'}, inplace=True)
    # change the paths to images from '/media/external_hdd_2/Datasets/SEWA/' to '/work/home/dsu/Datasets/SEWA/'
    labels['path'] = labels['path'].apply(lambda x: x.replace('/media/external_hdd_2/Datasets/SEWA/',
                                                                            '/work/home/dsu/Datasets/SEWA/'))
    # extract embeddings
    extract_embeddings_from_images_df(labels, model, device, preprocessing_functions, output_path,
                                        output_filename)

def extract_embeddings_AFEW_VA(path_to_data:str, model:torch.nn.Module, preprocessing_functions:List[Callable],
                                output_path:str, output_filename:str, device:torch.device):
    labels = pd.read_csv(os.path.join(path_to_data, 'labels.csv'))
    # filter out NAN values
    labels = labels[labels['frame_num'].notna()]
    # change columns name frame_num to path
    labels.rename(columns={'frame_num': 'path'}, inplace=True)
    # change the paths to images from '/media/external_hdd_1/Datasets/AFEW-VA/' to '/work/home/dsu/Datasets/AFEW-VA/'
    labels['path'] = labels['path'].apply(lambda x: x.replace('/media/external_hdd_1/Datasets/AFEW-VA/',
                                                                            '/work/home/dsu/Datasets/AFEW-VA/'))
    # extract embeddings
    extract_embeddings_from_images_df(labels, model, device, preprocessing_functions, output_path,
                                        output_filename)



def main():
    path_to_model_weights = "/work/home/dsu/tmp/radiant_fog_160.pth"
    preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size=240),
     EfficientNet_image_preprocessor()]
    model = construct_and_load_facial_model(path_to_model_weights)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    """# extract embeddings: EMOTIC
    print('Extracting embeddings from EMOTIC...')
    path_to_data = '/work/home/dsu/Datasets/EMOTIC/cvpr_emotic/cvpr_emotic/preprocessed/'
    output_path = '/work/home/dsu/tmp/Kamila/Embeddings/EMOTIC/emotional_model/'
    output_filename = 'emotic_radiant_fog_160'
    extract_embeddigs_EMOTIC(path_to_data, model, preprocessing_functions, output_path, output_filename, device)
    # extract embeddings: SEWA
    print('Extracting embeddings from SEWA...')
    path_to_data = '/work/home/dsu/Datasets/SEWA/preprocessed/'
    output_path = '/work/home/dsu/tmp/Kamila/Embeddings/SEWA/emotional_model/'
    output_filename = 'SEWA_radiant_fog_160'
    extract_embeddings_SEWA(path_to_data, model, preprocessing_functions, output_path, output_filename, device)"""
    # extract embeddings: AFEW-VA
    print('Extracting embeddings from AFEW-VA...')
    path_to_data = '/work/home/dsu/Datasets/AFEW-VA/AFEW-VA/AFEW-VA/preprocessed/'
    output_path = '/work/home/dsu/tmp/Kamila/Embeddings/AFEW-VA/emotional_model/'
    output_filename = 'AFEW-VA_radiant_fog_160'
    extract_embeddings_AFEW_VA(path_to_data, model, preprocessing_functions, output_path, output_filename, device)


if __name__ == '__main__':
    main()