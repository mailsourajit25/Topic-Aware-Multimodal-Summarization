'''
This file contains code to chunk similar/dissimilar image features for given list of url hashes and csv containing list of similar/dissimilar image paths
'''
import pandas as pd
import numpy as np
import time
import os
# os.chdir(os.path.join(os.getcwd(),"project","code","old_dsc_msmo"))

def form_chunks(out_dir, embedding_dir, prefix, col_name, all_df, url_hash_list, max_sim_img_num):
    outfile_path_prefix = os.path.join(out_dir,prefix)
    current_chunk_size=0
    chunk = []
    chunk_num = 0
    loop_count = 0
    for u_hash in url_hash_list:
        loop_count = loop_count + 1
        current_chunk_size+=1
        sim_files_path = all_df.loc[u_hash][col_name].split(",")
        unique_sim_hash = {}
        
        
        for p in sim_files_path:
            name_split = os.path.basename(p).split("_")
            hsh = name_split[0]
            count = int(name_split[1])
            unique_sim_hash[hsh] = count
            
        
        
        sample_embd = []
        rem_img = max_sim_img_num

        for uhsh in unique_sim_hash:
            loaded_embd = np.load(os.path.join(embedding_dir,uhsh+".npy"))
            img_count = unique_sim_hash[uhsh]
            sample_embd.append(loaded_embd[:,:img_count,:])
                
        concat_embed = np.concatenate(sample_embd,axis =1)
        assert concat_embed.shape[1]==max_sim_img_num, f"{u_hash} - concatenated images not equal to {max_sim_img_num}"
        chunk.append(concat_embed)
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
    csv_path = "../old_dsc_msmo/datasets/train_details(TT).csv"
    all_df = pd.read_csv(csv_path)
    all_df.set_index("URL_Hash", drop=False, inplace=True)
    url_hash_list = all_df["URL_Hash"].values

    CHUNK_SIZE = 1000
    DATA_PATH = "../../data"
    mode = "train"
    prefix = "features"
    embedding_dir = os.path.join(DATA_PATH,"embeddings/vgg19/"+mode)
    # out_dir = os.path.join(DATA_PATH,"dsc_msmo/finished_files/chunked_files/new_dissim_features_fc7")
    out_dir = os.path.join(DATA_PATH,"dsc_msmo/finished_files/chunked_files/new_chunk_sim_features_fc7/USE_TT/")
    
    col_name = "SIM_TT_USE_Path"
    # col_name = "DISSIM_Path"
    max_sim_img_num = 10
    if not os.path.exists(out_dir):
        print(f"Creating {out_dir}")
        os.makedirs(out_dir)
    print(f"---Storing similar embeddings in {out_dir}")
    # time.sleep(20)
    form_chunks(out_dir, embedding_dir, prefix, col_name, all_df, url_hash_list, max_sim_img_num)