import tensorflow as tf
import os
import time
from datetime import datetime
import h5py
import numpy as np
from collections import namedtuple 
import argparse



def vgg_feature_extractor():
    print("Preparing VGG19 Feature extractor...")
    with tf.device('cpu:0'):
        vgg19 = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
        vgg19.trainable = False
        #Extracting fc2 layer output
        vgg_output = [vgg19.get_layer(vgg19.layers[-2].name).output]
        model = tf.keras.Model([vgg19.input], vgg_output)
    print("VGG19 Feature extractor ready")
    return model

def load_img(fname,preprocessor):
    img=tf.io.read_file(fname)
    img=tf.io.decode_jpeg(img,3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img=tf.image.resize(img,(224,224))
    img=preprocessor(img)
    return fname, img

def show_rem_time(time_in_min):
    if time_in_min<60:
        return "{:.2f} min".format(time_in_min)
    else:
        hours=time_in_min/60
        rem_time=time_in_min%60
        return "{} hr {:.2f} min".format(int(hours),rem_time)


def get_features_dataset(image_dataset_batched,num_images,feature_extractor,layer_dim):
    print("Extracting Features from the images")
    num_batches=tf.data.experimental.cardinality(image_dataset_batched).numpy()
    features_dataset={"filenames":['' * num_images]}
    layer_shape=(num_images,layer_dim)
    features_dataset[layer_name] = np.zeros(layer_shape, np.float32)
    batch_idx=0
    #b[0] : filenames and b[1]: images
    for b in image_dataset_batched:
        t1=time.time()
        start=batch_idx*batch_size
        end=start+batch_size
        b_fnames=b[0].numpy()
        b_images=b[1]
        #Extracting features
        with tf.device('cpu:0'):
            outputs=feature_extractor(b_images)
        features_dataset[layer_name][start:end] = outputs
        features_dataset['filenames'][start:end]=b_fnames
        t2=time.time()
        time_per_batch=float(t2-t1)
        remaining_batches=(num_batches-(batch_idx+1))
        rem_time=show_rem_time((remaining_batches*time_per_batch)/60)
        print("[{}] Batch {:04d}/{:04d}, Batch Size = {}, Time/Batch = {:.2f}s, Rem Time = {}".format(
                datetime.now().strftime("%d/%m %H:%M"), batch_idx+1,
                num_batches, batch_size,time_per_batch, rem_time
            ))
        batch_idx+=1
    sort_features_dataset(features_dataset,layer_name)
    print("Feature Extraction complete")
    return features_dataset

def sort_features_dataset(features_dataset,layer_name):
    '''
    When more than one preprocessing thread is used the feature_dataset is
    not sorted according to alphabetical order of filenames. This function
    sorts the dataset in place so that filenames and corresponding fetaures
    are sorted by its filename. Note: sorting is in-place.

    :param features_dataset: dict, containting filenames and all features
    :param layer_name 
    :return:
    '''
    indices = np.argsort(features_dataset['filenames'])
    features_dataset['filenames'].sort()
    # Apply sorting to features for each image
    features_dataset[layer_name] = features_dataset[layer_name][indices]


def write_hdf5(out_filename, layer_name, features_dataset):
    '''
    Writes features to HDF5 file.
    :param out_filename: str, output file name (with path) 
    :param layer_names: list of str, layer names
    :param feature_dataset: dict, containing features[layer_names] = vals
    :return:
    '''
    print("Writing extracted features to H5 file")
    with h5py.File(out_filename, "w") as hf:
        hf.create_dataset("filenames", data=features_dataset['filenames'])
        hf.create_dataset(layer_name, data=features_dataset[layer_name], dtype=np.float32)
    print("Writing to H5 file complete")






def get_images_dataset_batched(image_dir,batch_size,preprocessor):
    print("Fetching batched image dataset..")
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    filenames_dataset=tf.data.Dataset.list_files(os.path.join(image_dir,'*.jpg'))
    image_dataset=filenames_dataset.map(lambda fname: load_img(fname,preprocessor),num_parallel_calls=AUTOTUNE)
    image_dataset_batched=image_dataset.batch(batch_size)
    num_images=tf.data.experimental.cardinality(filenames_dataset).numpy()
    image_dataset_batched = image_dataset_batched.cache().prefetch(buffer_size=AUTOTUNE)
    print("Batched image dataset Fetched..")
    return image_dataset_batched,num_images


_network=namedtuple("Attributes",['layer_name','layer_dim','preprocessor','feature_extractor'])

networks_details={
    "vgg_19":_network(
            "vgg_19/fc7",
            4096,
            tf.keras.applications.vgg19.preprocess_input,
            vgg_feature_extractor
        )
}



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TensorFlow feature extractor")
    parser.add_argument("--network", dest="network_name", type=str, required=True, help="model name, e.g. 'vgg_19'")
    parser.add_argument("--image_dir", dest="image_dir", type=str, required=True, help="path to directory containing images")
    parser.add_argument("--out_file", dest="out_file", type=str, default="./features.h5", help="path to save features (HDF5 file)")
    # parser.add_argument("--layer_names", dest="layer_names", type=str, required=True, help="layer names separated by commas")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="batch size for generating embeddings")
    args = parser.parse_args()
    
    batch_size=args.batch_size
    image_dir=args.image_dir
    out_file=args.out_file
    network=networks_details[args.network_name]
    
    layer_name=network.layer_name
    layer_dim=network.layer_dim
    preprocessor=network.preprocessor
    feature_extractor=network.feature_extractor()

    image_dataset_batched,num_images = get_images_dataset_batched(image_dir,batch_size,preprocessor)
    features_dataset=get_features_dataset(image_dataset_batched,num_images,feature_extractor,layer_dim)
    write_hdf5(out_file, layer_name, features_dataset)


