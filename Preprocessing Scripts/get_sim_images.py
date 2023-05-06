import glob
import os
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
import time
import pickle
# os.chdir(os.path.join(os.getcwd(), "project/code/preprocessing_scripts/"))


# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)





def embedding_generator(df,domain, embd_dir, embd_type):
    '''Embedding Generator'''
    # embd_type can be = title, img, caption
    df_domain = df[df["Domain"]==domain]
    for ind,row in df_domain.iterrows():
        url_hash = row["URL_Hash"]
        embd_path = os.path.join(embd_dir, embd_type, f"{url_hash}.npy")
        embd = np.load(embd_path)
        num_img = row["Img_Num"]
        output = {
            "img_num" : num_img,
            "embd" : embd,
            "hash" : url_hash
        }
        yield output


def get_datasets(df, domain, batch_size, embd_type, k):
    '''Returns bacthed datasets for comparision'''
    embd_dataset = tf.data.Dataset.from_generator(lambda: embedding_generator(df, domain,embedding_dir, embd_type),
                                                    output_types={
                                                        "img_num" : tf.int32,
                                                        "embd" : tf.float32,
                                                        "hash" : tf.string
                                                        })
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    batched_embd_dataset = embd_dataset.batch(batch_size,num_parallel_calls=AUTOTUNE)
    batched_embd_dataset= batched_embd_dataset.prefetch(buffer_size=AUTOTUNE)
    batched_embd_comp_dataset = embd_dataset.batch(batch_size,num_parallel_calls=AUTOTUNE, drop_remainder=True)
    batched_embd_comp_dataset= batched_embd_comp_dataset.shuffle(buffer_size = k, reshuffle_each_iteration=True)
    return batched_embd_dataset, batched_embd_comp_dataset


def get_sim_data(row, img_dict):
    '''Row-wise apply store similar image data within dict'''
    row["SIM_TT_USE_Path"] = img_dict[row["URL_Hash"]][0][:-1]
    row["SIM_TT_USE_Score"] = img_dict[row["URL_Hash"]][1][:-1]
    return row

def store_dict_in_csv(df, img_dict, csv_path):
    '''Saves similar images dictionary into csv'''
    print("Storing in CSV")
    sim_img_df = df.apply(lambda row : get_sim_data(row, img_dict),axis=1)
    sim_img_df.to_csv(csv_path,index=False)
    print("CSV Saved")
    

def find_sim_scores(src, comp):
    '''Calculates cosine similarity scores'''
    sim_scores = tf.reduce_sum(src * comp, axis=-1)
    src_norm = tf.norm(src, axis=-1)
    comp_norm = tf.norm(comp, axis=-1)
    norm_mul = src_norm * comp_norm
    sim_scores = sim_scores / tf.where( tf.equal(norm_mul,0.0) ,1.0, norm_mul)
    # sim_scores = tf.clip_by_value(sim_scores, -1.0 , 1.0)
    return sim_scores 


def find_similar_images(domain, batch_size, df, embd_type, rand_num_batch, k, sim_img_dict):
    '''Finds similar images for a given domain'''
    #getting embedding datasets
    batched_embd_dataset, batched_embd_comp_dataset = get_datasets(df,domain,batch_size, embd_type, k)
    i=0
    max_sim_img_num = k
    #Incrementing k by 1 so that atleast k samples can be selected
    k = k+1
    for batch_src in batched_embd_dataset:
        # print(i)
        # if (i+1) not in range(15,16):
        #     i+=1
        #     continue
        batch_src_ext = tf.repeat(batch_src['embd'],batch_size,axis = 0) # shape (cur_batch_size * batch_size, embd_dim)
        cur_batch_size = batch_src['embd'].shape[0]
        j=0
        topk_scores = None
        topk_hash = None
        topk_img_num = None
        prev_topk_scores= None
        prev_topk_hash = None
        prev_topk_img_num = None
        batch_time_start = time.time()
        for batch_comp in batched_embd_comp_dataset.take(rand_num_batch):
            # if batch_comp["img_num"].shape[0] < batch_size:
            #     continue
            batch_com_ext = tf.tile(batch_comp['embd'],[cur_batch_size,1]) #shape == (cur_batch_size*batch_size, embd_dim)
            batch_com_hash_ext = tf.repeat([batch_comp["hash"]], cur_batch_size, axis=0) #shape == (cur_batch_size, batch_size)
            batch_com_inum_ext = tf.repeat([batch_comp["img_num"]], cur_batch_size, axis=0) #shape == (cur_batch_size, batch_size)
            scores = find_sim_scores(batch_src_ext, batch_com_ext)
            
            # if i==j:
            #     scores = tf.where(tf.equal(scores,1.0),0.0, scores)
            
            scores = tf.reshape(scores, [-1, batch_size]) #shape (cur_batch_size, batch_size)
            
            if(j>0):
                #For 2nd iteration onwards we append previously found topk values 
                scores = tf.concat([scores, prev_topk_scores.values], axis=1) #shape (cur_batch_size , batch_size + k)
                batch_com_hash_ext = tf.concat([batch_com_hash_ext, prev_topk_hash], axis=1) #shape (cur_batch_size , batch_size + k)
                batch_com_inum_ext = tf.concat([batch_com_inum_ext, prev_topk_img_num], axis=1) #shape (cur_batch_size , batch_size + k)

            topk_scores = tf.math.top_k(scores, k)
            topk_hash = tf.gather(batch_com_hash_ext,indices=topk_scores.indices, batch_dims=1, axis=1) # shape == (batch_size , k)
            topk_img_num = tf.gather(batch_com_inum_ext,indices=topk_scores.indices, batch_dims=1, axis=1) # shape == (batch_size , k)
            
            #Assigning previous values
            prev_topk_scores, prev_topk_hash, prev_topk_img_num = topk_scores, topk_hash, topk_img_num
            print(f"Comparing done - domain : {domain} , batch : {i+1} with batch : {j+1} ")
            j+=1
        

        print(f"Forming dictionary for batch {i+1} ..")
        for s_num in tf.range(cur_batch_size):
            img_sum=0
            sim_imgs = ""
            sim_scores = ""
            #sample hash
            s_batch_src_hash = batch_src['hash'][s_num]
            for k_num in tf.range(k):
                if img_sum >= max_sim_img_num:
                    #if number of images exceeds k then break
                    break
                if tf.equal(topk_hash[s_num][k_num],s_batch_src_hash):
                    #if same element found
                    continue
                # Number of images in sample s_num in batch and similar sample number k_num out of topk similar samples
                sk_img_num = topk_img_num[s_num][k_num].numpy()
                # Remaining images
                sk_img_num = min(sk_img_num,max_sim_img_num-img_sum)
                sim_scores = sim_scores + str(round(topk_scores.values[s_num][k_num].numpy(),3)) + ","
                sim_img_url_hash = topk_hash[s_num][k_num].numpy().decode()
                sim_img_folder_path = df.loc[sim_img_url_hash]["Folder_Path"]
                for img_idx in range(sk_img_num):    
                    sim_img_path = sim_img_folder_path + "/" + sim_img_url_hash + f"_{img_idx+1} ,"
                    sim_imgs = sim_imgs + sim_img_path
                img_sum = img_sum + sk_img_num
            assert img_sum==max_sim_img_num, f"There must be 10 similar images per sample. Given sample has {img_sum} images"
            sim_img_dict[s_batch_src_hash.numpy().decode()] = (sim_imgs, sim_scores) 
        batch_run_time = round(time.time() - batch_time_start, 2 )
        print(f"Similar images found : domain : {domain} , batch - {i+1} , batch_size - {batch_size} , time - {batch_run_time} sec")
        i+=1

def find_all_sim_images(batch_size, df, embd_type, rand_num_batch,k,out_csv_path):
    '''Finds similar images for all domains'''
    sim_img_dict={}
    for domain in df["Domain"].unique():
        # print(domain)
        # if domain != "sport":
        #     continue
        find_similar_images(domain, batch_size, df, embd_type, rand_num_batch,k, sim_img_dict)
        try:
            dict_file = open('../old_dsc_msmo/datasets/new_sim_img_TT_dict', 'wb')
            pickle.dump(sim_img_dict, dict_file)
            dict_file.close()
        except:
            print("Something went wrong")
    store_dict_in_csv(df,sim_img_dict,out_csv_path)


if __name__ == "__main__":
    DATA_PATH="../../data/"
    embedding_dir=os.path.join(DATA_PATH,"embeddings","use")
    all_df=pd.read_csv("../old_dsc_msmo/datasets/new_train_details(TT).csv")
    all_df.set_index("URL_Hash", drop=False, inplace=True)
    embd_type = "title"
    #batch_size * rand_num_batch = number of random examples we are comparing per sample
    batch_size=1000
    k=10 #for taking top k samples
    rand_num_batch = 20 # Number of random batches to be considered for comparing
    out_csv_path = "../old_dsc_msmo/datasets/new_train_details(TT).csv"
    find_all_sim_images(batch_size, all_df, embd_type, rand_num_batch,k, out_csv_path)
