import glob
import os
import shutil









if __name__=="__main__":
    output_path = "/work/home/dsu/Datasets/"
    data_path = "/media/external_hdd_2/Datasets/"
    template = os.path.join(data_path, "FER_plus","images")
    datasets = glob.glob(template)
    print("Found datasets:", datasets)

    for dataset in datasets:
        print("Working on: ", dataset)
        # move all files from dataset dir to output_path
        adding_to_output = dataset.split(os.path.sep)[-2:]
        final_output = os.path.join(output_path, *adding_to_output)
        print("Output path will be: ", final_output)
        #os.makedirs(final_output, exist_ok=True)
        # copy all files and subdirectories recursively from dataset to final_output
        shutil.copytree(dataset, final_output, dirs_exist_ok = True)

