import numpy as np
import h5py
import os
import hashlib
import glob
import re



path="../../data/msmo/features_fc7/backup"
hf = h5py.File(os.path.join(path,'features1.h5'), 'r')
hf.keys()
hf['filenames'][:10]
### Getting URL Hashes
urls_path=os.path.join("url_lists","all_train.txt")
all_urls=open(urls_path,'r')
def get_hash(url):
  url_b=url.encode('utf-8')
  return hashhex(url_b.strip())

#Hash function to hash the url to generate file name
def hashhex(s): 
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()
def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines
def get_url_hashes(url_list):
  return np.array([get_hash(url) for url in url_list])
url_list=read_text_file(urls_path)
url_hashes=get_url_hashes(url_list)
url_hashes[0]
CHUNK_SIZE=1000
def get_filenum(path):
    # for "../data/features3.h5" -> returns -> 3
    fname=os.path.split(path)[-1].split(".")[0]
    fnum=int(re.sub('\D', '', fname))
    return fnum

get_filenum("../data/features6.h5")

def extract_url_hash(path):
    path=path.decode('ascii')
    fname=path.split("/")[-1]
    url_hash=fname.split("_")[0]
    return url_hash

#For using this in numpy map
extract_url_hash_np=np.vectorize(extract_url_hash)
def form_chunk(out_files_dir,outfile_prefix,inp_files_dir,inpfile_prefix,url_hashes,network_name,layer_name):
    '''
    Forms chunks from given directory of h5 feature files
    inpfile_prefix : h5 filenames prefix. Ex for features3.h5, prefix is features
    '''
    if not os.path.isdir(out_files_dir):
        os.mkdir(out_files_dir)
    files_path=glob.glob(os.path.join(inp_files_dir,f"{inpfile_prefix}*.h5"))
#     files_path.sort(key=lambda f: get_filenum(f))
    #initializing chunk_features - stores CHUNK_SIZE no. of features
    chunk_features=[]
    #dynamically updates chunk size
    current_chunk_size=0
    samples_covered=0
    total_samples_covered=0
    chunk_num=0
    outfile_path_prefix = os.path.join(out_files_dir,outfile_prefix)
    for file in files_path:
        hf=h5py.File(file, 'r')
        features_embeds=hf[network_name][layer_name]
        #converting to numpy array
        filenames = hf['filenames'][()]
        #extracting url_hashes from filenames
        features_url_hash=extract_url_hash_np(filenames)
        #forming dictionary of url_hash and indices of the images belonging to that hash
        uhash_idx_dict={}
        for i in range(features_url_hash.shape[0]):
            if features_url_hash[i] in uhash_idx_dict:
                uhash_idx_dict[features_url_hash[i]].append(i)
            else:
                uhash_idx_dict[features_url_hash[i]]=[i]
        
        for u_hash in url_hashes[total_samples_covered:]:
            #if url_hash is not present in current file
            if u_hash not in uhash_idx_dict:
                total_samples_covered=samples_covered
                break
            chunk_features.append(np.concatenate(features_embeds[uhash_idx_dict[u_hash]],axis=1))
            current_chunk_size+=1
            samples_covered+=1
            if current_chunk_size==CHUNK_SIZE:
                np.savez_compressed("%s_%03d"%(outfile_path_prefix,chunk_num),*chunk_features)
                print(f"Saving chunk number - {chunk_num}")
                chunk_features=[]
                current_chunk_size=0
                chunk_num+=1
    #saving the left over features as final chunk
    if current_chunk_size!=0:
        np.savez_compressed("%s_%03d"%(outfile_path_prefix,chunk_num),*chunk_features)
        print(f"Saving chunk number - {chunk_num}")
    print("All chunks saved")
        
path="../../data/msmo/"
# form_chunk(
#     out_files_dir=os.path.join(path,"finished_files","chunk_fc7"),
#     outfile_prefix="features"
#     inp_files_dir=os.path.join(path,"features_fc7"),
#     inpfile_prefix="features"
#     url_hashes=url_hashes,
#     network_name="vgg19",
#     layer_name="fc7"
# )
test=np.load(os.path.join(path,"finished_files","chunk_fc7","features_293.npz"))
len(test.files)
test["arr_0"].shape
urls_path=os.path.join("url_lists","all_val.txt")
all_urls=open(urls_path,'r')
url_list=read_text_file(urls_path)
url_hashes=get_url_hashes(url_list)
path="../../data/msmo/"
form_chunk(
    out_files_dir=os.path.join(path,"finished_files","chunk_fc7"),
    outfile_prefix="valid_features",
    inp_files_dir=os.path.join(path,"features_fc7"),
    inpfile_prefix="valid_features",
    url_hashes=url_hashes,
    network_name="vgg_19",
    layer_name="fc7"
)
urls_path=os.path.join("url_lists","all_test.txt")
all_urls=open(urls_path,'r')
url_list=read_text_file(urls_path)
url_hashes=get_url_hashes(url_list)
path="../../data/msmo/"
form_chunk(
    out_files_dir=os.path.join(path,"finished_files","chunk_fc7"),
    outfile_prefix="test_features",
    inp_files_dir=os.path.join(path,"features_fc7"),
    inpfile_prefix="test_features",
    url_hashes=url_hashes,
    network_name="vgg_19",
    layer_name="fc7"
)

glob.glob(os.path.join(os.path.join(path,"features_fc7"),f"valid_features*.h5"))
hfile=h5py.File('../../data/msmo/features_fc7/valid_features.h5', 'r')
hfile.keys()