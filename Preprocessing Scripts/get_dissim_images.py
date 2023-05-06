import pandas as pd
import numpy as np



def find_dissim_images(df, df_dict, url_hash_list, max_dissim_images):
    dissim_img_paths = {}
    loop_count = 0
    for u_hash in url_hash_list:
        loop_count+=1
        out_domain_df = df_dict[df.loc[u_hash]["Domain"]]
        random_out_domain_samples = out_domain_df.sample(max_dissim_images)
        samp_dissim_imgs = ""
        rem_img = max_dissim_images
        for index, row in random_out_domain_samples.iterrows():
            if rem_img <= 0:
                break
            if row["Img_Num"] > rem_img:
                for img_idx in range(rem_img):
                    samp_dissim_imgs = samp_dissim_imgs + row["Folder_Path"] + "/" + row["URL_Hash"] + f"_{img_idx+1} ,"
                rem_img = 0
                break
            else:
                for img_idx in range(row["Img_Num"]):
                    samp_dissim_imgs = samp_dissim_imgs + row["Folder_Path"] + "/" + row["URL_Hash"] + f"_{img_idx+1} ,"
                rem_img = rem_img - row["Img_Num"]
        assert rem_img==0, f"10 Dissimilar images not chosen. {rem_img} images remaining"
        dissim_img_paths[u_hash] = samp_dissim_imgs
        if loop_count%20 == 0:
            print(f"Processed {loop_count} urls")
    return dissim_img_paths


def get_sim_data(row, img_dict):
    row["DISSIM_Path"] = img_dict[row["URL_Hash"]][:-1]
    return row

def store_dict_in_csv(df, img_dict, csv_path):
    print("Storing in CSV")
    sim_img_df = df.apply(lambda row : get_sim_data(row, img_dict),axis=1)
    sim_img_df.to_csv(csv_path,index=False)
    print("CSV Saved")

if __name__ == '__main__':
    csv_path = "../old_dsc_msmo/datasets/train_backup/train_details(TT).csv"
    all_df = pd.read_csv(csv_path)
    all_df.set_index("URL_Hash", drop=False, inplace=True)

    df_dict = {}
    for domain in all_df["Domain"].unique():
        df_dict[domain] = all_df[all_df["Domain"]!=domain]

    url_hash_list = all_df["URL_Hash"].values
    max_dissim_images = 10
    dissim_img_dict = find_dissim_images(all_df, df_dict, url_hash_list, max_dissim_images)
    out_csv_path = "../old_dsc_msmo/datasets/train_backup/train_details(TT).csv"
    store_dict_in_csv(all_df, dissim_img_dict, out_csv_path)