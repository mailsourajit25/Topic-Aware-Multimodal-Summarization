'''
This file contains code to chunk image features for given list of url hashes
'''

import pandas as pd
import numpy as np
import os

def form_chunks(out_dir, embedding_dir, prefix, url_hash_list):
    outfile_path_prefix = os.path.join(out_dir,prefix)
    current_chunk_size=0
    chunk = []
    chunk_num = 0
    loop_count = 0
    for u_hash in url_hash_list:
        loop_count = loop_count + 1
        current_chunk_size+=1
        sample_embd = np.load(os.path.join(embedding_dir,f"{u_hash}.npy"))
        chunk.append(sample_embd)
        if current_chunk_size == CHUNK_SIZE :
            np.savez_compressed("%s_%03d"%(outfile_path_prefix,chunk_num),*chunk)
            print(f"Chunk No. - {chunk_num} saved")
            chunk_num = chunk_num + 1
            current_chunk_size = 0
            chunk = []
        print(f"Processed {loop_count} urls")
    if len(chunk)!=0:
        np.savez_compressed("%s_%03d"%(outfile_path_prefix,chunk_num),*chunk)
        print(f"Chunk No. - {chunk_num} saved")



if __name__ == '__main__' :
    csv_path = "datasets/test_details.csv"
    all_df = pd.read_csv(csv_path)
    all_df.set_index("URL_Hash", drop=False, inplace=True)
    url_hash_list = all_df["URL_Hash"].values



    CHUNK_SIZE = 1000
    DATA_PATH = "../../data"
    mode = "test"
    prefix = "features"
    if mode!= "train":
        prefix = mode + "_" + "features"
    embedding_dir = os.path.join(DATA_PATH,"embeddings/vgg19/"+mode)
    out_dir = os.path.join(DATA_PATH,"dsc_msmo/finished_files/chunked_files/chunk_features_fc7")
    print("Mode",mode)
    print("Prefix",prefix)
    max_sim_img_num = 10
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    form_chunks(out_dir, embedding_dir, prefix, url_hash_list)