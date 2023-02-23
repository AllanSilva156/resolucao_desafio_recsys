
import argparse
import pandas as pd
import numpy as np
#from transformers import pipeline
from tqdm import tqdm

import json
from torchvision import datasets, transforms, models
import torch

import matplotlib.pyplot as plt

import cv2
import glob
import os
import numpy as np


def load_dataset(json_file_img: str):
    """
    Load the dataset from a json file.
    """
    with open(json_file_img, 'r') as file:
        data = []
        for line in file.readlines():
            data.append(pd.Series(json.loads(line)))

    df_img = pd.concat(data, axis=1).T
    return df_img


def load_model():
    """
    Load the model from https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html

    """
    model = models.resnet50(pretrained=True)
    model_no_top = torch.nn.Sequential(
        *(list(model.children())[:-1]))  # without classification head

    return model_no_top


def resize_images(files, destination):
    try:                                        # making the destination folder
        if not os.path.exists(destination):
            os.makedirs(destination)
    except OSError:
        print('Error while creating directory')

    image_no = 0
    resolution = (224, 224)  # use your desired resolution

    # insert your input image folder directory here
    for img_path in tqdm(glob.glob(files)):
        # if the folder contains various types of images rather than only .jpg therefore use *.* instead of *.jpg (but make sure the folder contains only images)
        img = cv2.imread(img_path)

        if img is None:  # checking if the read image is not a NoneType
            continue

        # the image will be zoomed in or out depending on the resolution
        img = cv2.resize(img, resolution)

        cv2.imwrite(destination + img_path.split('\\')[-1], img)

        image_no = image_no + 1


def get_embeddings(model, df: pd.DataFrame, column: str, origin: str, detination: str):
    """
    Get the embedding for a column.
    """
    pbar = tqdm(range(df.shape[0]))
    count_errors = 0
    batch = []
    index_errors = []
    index_emb = []

    data = []
    for i in pbar:
        pbar.set_description(f"Erros: {count_errors}")
        pbar.refresh()
        file_name = df.loc[i, column]

        if len(batch) % 64 == 0 and len(batch) > 0:
            inputs = torch.cat(batch, axis=0)
            embeddings = model(inputs).squeeze().detach().numpy()
            data.append(embeddings)
            df.loc[index_emb, ['embs']] = [
                str(list(emb)) for emb in embeddings]
            df.loc[index_errors, ['embs']] = str(list(np.zeros(2048)))
            batch = []
            index_errors = []
            index_emb = []

        if i % 10000 == 0:
            df.to_csv(detination, index=False)

        try:
            photo = plt.imread(
                f'{origin}/{file_name}.jpg')
            input = torch.Tensor(photo.copy()).permute(2, 0, 1).unsqueeze(0)
            index_emb.append(i)
            batch.append(input)
        except:
            count_errors += 1
            index_errors.append(i)
            continue
    if len(batch) > 0:
        inputs = torch.cat(batch, axis=0)
        embeddings = model(inputs).squeeze().detach().numpy()
        data.append(embeddings)
        df.loc[index_emb, ['embs']] = [
            str(list(emb)) for emb in embeddings]
        df.loc[index_errors, ['embs']] = str(list(np.zeros(2048)))
        batch = []
        index_errors = []
        index_emb = []
    return df


def mean_emb(x):
    return x.apply(lambda x: np.array(eval(x))).mean()


def export_dataset(df: pd.DataFrame, emb_column: str, output_file: str):
    """
    Export the embeddings to a csv file.
    """
    np.savetxt(output_file+'/embeddings.txt',
               np.stack(df[emb_column]), delimiter='\t')
    df.drop(emb_column, axis=1).to_csv(
        output_file+"/metadados.csv", sep="\t", index=False)


if __name__ == '__main__':
    """
    Extract the embeddings from a dataset - baseline code.

    Params:

    model_base: The model base to extract the embeddings.
    csv_file: The csv file to extract the embeddings.
    output_path: The output path to save the embeddings and metadata.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('json_file', type=str, help='The json file',)
    parser.add_argument('output_path', type=str, help='Output Path',)

    args = parser.parse_args()

    # Load Dataset
    print("\n\nLoad Dataset...")
    df_img = load_dataset(args.json_file)
    print(df_img.head())

    # Load Model
    print("\n\nLoad Transform Model...")
    model = load_model()
    print(model)

    destination = 'data\\yelp_dataset\\resized_photos\\'
    files = r'data\\yelp_dataset\\photos/*.jpg'
    resize_images(files, destination)

    # Extract Embeddings
    print("\n\nExtract Embeddings...")
    df_img = get_embeddings(
        model,
        df_img,
        "photo_id",
        origin='data/yelp_dataset/resized_photos',
        detination='data/yelp_dataset/dataset_embedding_2.csv'
    )  
    df_img = df_img.rename({'embeddings': 'embs'}, axis=1)
    # Completes null values ​​with null embeddings
    df_img['embs'] = df_img['embs'].fillna(str(list(np.zeros(2048))))

    print("Calculating averange embeddings...")
    # Averages the embeddings by business_id
    df_result_mean_emb = df_img.groupby('business_id').agg({'embs': mean_emb})
    df_result_mean_emb.loc[df_result_mean_emb['embs'].isna(), ['embs']] = df_result_mean_emb.loc[
        df_result_mean_emb['embs'].isna(),
        ['embs']
    ].applymap(lambda x: np.zeros(2048))

    print(df_result_mean_emb.head())

    # For businesses without a photo, assign null embedding
    df_bus = pd.read_csv(
        'data\\yelp_dataset\\yelp_academic_dataset_business.csv')
    df_export = df_bus.merge(
        df_result_mean_emb.reset_index(), on='business_id', how='left')
    df_export.loc[df_export['embs'].isna(), ['embs']] = df_export.loc[
        df_export['embs'].isna(),
        ['embs']
    ].applymap(lambda x: np.zeros(2048))

    # Export Dataset
    print("\n\nExtract Dataset...")
    export_dataset(
        df_result_mean_emb[['business_id', 'embs']], "embs", args.output_path
    )

    print("\n\nDone! :)")
