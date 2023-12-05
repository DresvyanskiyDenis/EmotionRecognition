import os.path
from functools import partial

import torch
from torchinfo import summary

from feature_extraction.embeddings_extraction_torch import EmbeddingsExtractor
from pytorch_utils.models.CNN_models import Modified_EfficientNet_B1
from pytorch_utils.models.input_preprocessing import resize_image_saving_aspect_ratio, EfficientNet_image_preprocessor
from src.training.dynamic.facial.data_preparation import load_all_dataframes


def create_and_load_extractor():
    path_to_weights_base_model = "/work/home/dsu/tmp/radiant_fog_160.pth"
    extractor = Modified_EfficientNet_B1(embeddings_layer_neurons=256, num_classes=8,
                                               num_regression_neurons=2)
    extractor.load_state_dict(torch.load(path_to_weights_base_model))
    # cut off last two layers of base model
    extractor = torch.nn.Sequential(*list(extractor.children())[:-2])
    # freeze base model
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    return extractor


def main():
    # params
    output_path = "/work/home/dsu/Datasets/Emo_Datasets_Embeddings/"
    path_to_weights_base_model = "/work/home/dsu/tmp/radiant_fog_160.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocessing_functions = [partial(resize_image_saving_aspect_ratio, expected_size = 240),
                                   EfficientNet_image_preprocessor()]
    extractor = create_and_load_extractor()
    summary(extractor, (2,3, 240, 240))
    # create output paths
    os.makedirs(os.path.join(output_path, 'AFEW-VA'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'RECOLA'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'SEMAINE'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'SEWA'), exist_ok=True)

    # load data
    train, dev, test = load_all_dataframes(seed = 101095)
    # divide data into different datasets (they are concatenated now)
    # we need to divide it depending on the 'path' columns
    # we have there 4 datasets: 'AFEW-VA', 'RECOLA', 'SEMAINE', 'SEWA'
    # train
    train_afew = train[train['path'].str.contains('AFEW-VA')]
    train_recola = train[train['path'].str.contains('RECOLA')]
    train_semaine = train[train['path'].str.contains('SEMAINE')]
    train_sewa = train[train['path'].str.contains('SEWA')]
    # dev
    dev_afew = dev[dev['path'].str.contains('AFEW-VA')]
    dev_recola = dev[dev['path'].str.contains('RECOLA')]
    dev_semaine = dev[dev['path'].str.contains('SEMAINE')]
    dev_sewa = dev[dev['path'].str.contains('SEWA')]
    # test
    test_afew = test[test['path'].str.contains('AFEW-VA')]
    test_recola = test[test['path'].str.contains('RECOLA')]
    test_semaine = test[test['path'].str.contains('SEMAINE')]
    test_sewa = test[test['path'].str.contains('SEWA')]
    embeddings_extractor = EmbeddingsExtractor(model=extractor, device=device,
                 preprocessing_functions = preprocessing_functions, output_shape = 256)
    # extract embeddings AFEW-VA
    final_output_path = os.path.join(output_path, 'AFEW-VA', 'afew_va_train.csv')
    embeddings_extractor.extract_embeddings(data=train_afew, batch_size = 64, num_workers = 16,
                           output_path = final_output_path, verbose = True)
    final_output_path = os.path.join(output_path, 'AFEW-VA', 'afew_va_dev.csv')
    embeddings_extractor.extract_embeddings(data=dev_afew, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)
    final_output_path = os.path.join(output_path, 'AFEW-VA', 'afew_va_test.csv')
    embeddings_extractor.extract_embeddings(data=test_afew, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)
    # extract embeddings RECOLA
    final_output_path = os.path.join(output_path, 'RECOLA', 'recola_train.csv')
    embeddings_extractor.extract_embeddings(data=train_recola, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)
    final_output_path = os.path.join(output_path, 'RECOLA', 'recola_dev.csv')
    embeddings_extractor.extract_embeddings(data=dev_recola, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)
    final_output_path = os.path.join(output_path, 'RECOLA', 'recola_test.csv')
    embeddings_extractor.extract_embeddings(data=test_recola, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)
    # extract embeddings SEMAINE
    final_output_path = os.path.join(output_path, 'SEMAINE', 'semaine_train.csv')
    embeddings_extractor.extract_embeddings(data=train_semaine, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)
    final_output_path = os.path.join(output_path, 'SEMAINE', 'semaine_dev.csv')
    embeddings_extractor.extract_embeddings(data=dev_semaine, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)
    final_output_path = os.path.join(output_path, 'SEMAINE', 'semaine_test.csv')
    embeddings_extractor.extract_embeddings(data=test_semaine, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)
    # extract embeddings SEWA
    final_output_path = os.path.join(output_path, 'SEWA', 'sewa_train.csv')
    embeddings_extractor.extract_embeddings(data=train_sewa, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)
    final_output_path = os.path.join(output_path, 'SEWA', 'sewa_dev.csv')
    embeddings_extractor.extract_embeddings(data=dev_sewa, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)
    final_output_path = os.path.join(output_path, 'SEWA', 'sewa_test.csv')
    embeddings_extractor.extract_embeddings(data=test_sewa, batch_size = 64, num_workers = 16,
                            output_path = final_output_path, verbose = True)







if __name__ == '__main__':
    main()