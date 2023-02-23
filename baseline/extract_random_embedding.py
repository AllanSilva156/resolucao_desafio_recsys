
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def load_dataset(csv_file: str):
  """
  Load the dataset from a csv file.
  """
  df = pd.read_csv(csv_file)[["business_id", "name", "categories"]]
     
  return df

def get_embeddings(df: pd.DataFrame):
  """
  Get the embedding for a column.
  """
  data = []
  for i, row in tqdm(df.iterrows(), total=df.shape[0]):
      data.append(np.random.random(128))

  return data

def export_dataset(df: pd.DataFrame, emb_column: str, output_file: str):
  """
  Export the embeddings to a csv file.
  """
  if not os.path.exists(output_file):
      os.makedirs(output_file)

  np.savetxt(output_file+'/embeddings.txt', np.stack(df[emb_column]), delimiter='\t')
  df.drop(emb_column, axis=1).to_csv(output_file+"/metadados.csv", sep="\t", index=False)

if __name__ == '__main__':
  """
  Extract random embeddings from a dataset - baseline code.
  
  Params:
  
  csv_file: The csv file to extract the embeddings.
  output_path: The output path to save the embeddings and metadata.
  """

  parser = argparse.ArgumentParser()

  parser.add_argument('csv_file',type=str,help='The csv file',)
  parser.add_argument('output_path',type=str,help='Output Path',)

  args = parser.parse_args()

  # Load Dataset
  print("\n\nLoad Dataset...")
  df = load_dataset(args.csv_file)
  print(df.head()) 

  # Extract Embeddings
  print("\n\nExtract Embeddings...")
  df["embs"] = get_embeddings(df)
  df["embs"] = df["embs"].apply(np.array)
  print(df.head())

  #Exporta Dataset
  print("\n\nExtract Dataset...")

  export_dataset(df, "embs", args.output_path)

  print("\n\nDone! :)")