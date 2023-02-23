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
