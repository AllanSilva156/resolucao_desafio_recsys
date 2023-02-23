# resolucao_desafio_recsys
Repositório destinado ao registro da resolução do desafio de RecSys proposto como projeto final da disciplina de Aprendizado de Máquina Não Supervisionado ofertada para o Bacharelado de Inteligência Artificial (Universidade Federal de Goiás).

## Conversão de `.json` para `.csv`

O primeiro passo após a importação dos dados é a conversão dos arquivos do formato original `.json` para o formato de tabela `.csv`. O comando abaixo realiza este processo.

```bash
python scripts/json_to_csv_converter.py data/yelp_dataset/yelp_academic_dataset_business.json
```

## Experimentos

Para a reprodução dos experimentos, basta seguir o passo a passo abaixo.

### Geração de embeddings (texto)

O exemplo de comando abaixo irá gerar os embeddings dos dados textuais referentes ao nome, a localização e a quantidade de estrelas na avaliação dos estabelecimentos.

```bash
python experimentos/name_loc_stars.py bert-base-uncased data/dataset/yelp_dataset/yelp_academic_dataset_business.csv data/output/name_loc_stars_emb
```

Os arquivos resultantes são `metadados.csv` e `embeddings.txt`.

### Geração de embeddings (imagem)

O exemplo de comando abaixo irá gerar os embeddings das fotos disponíveis dos estabelecimentos.

```bash
python experimentos/extract_photo_embedding.py data/yelp_dataset/photos.json data/output/photos_emb
```

Os arquivos resultantes são `metadados.csv` e `embeddings.txt`.

### Junção dos embeddings

O arquivo `juntar_embeddings.py` possui a função de agregar todos os embeddings em um único arquivo final `embeddings.txt`. No código abaixo a única modificação a ser feita é a inserção dos arquivos a serem agregados dentro da variável `files_embeddings`.

```
import numpy as np

# Place the embeddings you want to join
files_embeddings = ['data/models/name_categ/embeddings.txt',
                    'data/models/resnet/embeddings.txt']
list_embeddings = []
for file in files_embeddings:
    list_embeddings.append(np.loadtxt(file))

concat_embs = np.concatenate(list_embeddings, axis=1)

# Put the folder path where the resulting embeddigs will be saved
output_file = 'data/models/name_categ_img'
np.savetxt(output_file+'/embeddings.txt', concat_embs, delimiter='\t')
```

### Avaliação

A avaliação dos embeddings gerados é feita a partir do motor de recomendação contido em `evaluation.py`. O comando a seguir repassa os embeddings ao motor de recomendação.

```bash
python evaluation/evaluation.py <embedding_path> <metadados_path>
```

As métricas são geradas em termos de `NDCG@5` e `NDCG@10`.
