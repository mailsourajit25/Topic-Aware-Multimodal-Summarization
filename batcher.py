import tensorflow as tf
import numpy as np
import glob
import struct
import data




def get_mask(actual_len, target_len):
    if actual_len>=target_len:
        return tf.ones([target_len],tf.float32)
    else:
        ones=tf.ones([actual_len],tf.float32)
        zeros=tf.zeros([target_len-actual_len],tf.float32)
        return tf.concat([ones,zeros],axis=0)



def raw_record_generator(data_path, img_feature_path, logger, sim_img_feature_path = None, dissim_img_feature_path = None, max_img_num=10):
    """Generates tf.Examples from data files.

        Binary data format: <length><blob>. <length> represents the byte size
        of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
        the tokenized article text and summary.

    Args:
        data_path:
        Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
        single_pass:
        Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.

    Yields:
        Deserialized tf.Example.
    """

    filelist = glob.glob(data_path) # get the list of datafiles
    feature_list = glob.glob(img_feature_path)
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    assert feature_list, ('Error: Empty feature_list at %s' % img_feature_path)

    filelist = sorted(filelist)
    feature_list = sorted(feature_list)

    sim_feature_list = glob.glob(sim_img_feature_path)
    dissim_feature_list = glob.glob(dissim_img_feature_path)
    assert sim_feature_list, ('Error: Empty feature_list at %s' % sim_img_feature_path)
    assert dissim_feature_list, ('Error: Empty feature_list at %s' % dissim_img_feature_path)
    sim_feature_list = sorted(sim_feature_list)
    dissim_feature_list = sorted(dissim_feature_list)
    # text_img_pairs = list(zip(filelist, feature_list, sim_feature_list, dissim_feature_list))

    

    for c_idx in range(len(filelist)):
        text_f = filelist[c_idx]
        img_f = feature_list[c_idx]
        reader = open(text_f, 'rb')
        chunk_img_arr = np.load(img_f)
        sim_img_f = sim_feature_list[c_idx]
        dissim_img_f = dissim_feature_list[c_idx]
        chunk_sim_img_arr = np.load(sim_img_f)
        chunk_dissim_img_arr = np.load(dissim_img_f)
        img_idx = 0
        img_num = len(list(chunk_img_arr))
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: break # finished reading this file
            if img_idx == img_num: break
            img_feature = chunk_img_arr['arr_{}'.format(img_idx)]
            img_feature = img_feature[:, :max_img_num, :]
            #For similar images
            #Dataset has max 10 similar and 10 dissimilar images per article
            sim_img_feature = chunk_sim_img_arr['arr_{}'.format(img_idx)]
            sim_img_feature = sim_img_feature[:, :max_img_num, :]
            #For dissimilar images
            dissim_img_feature = chunk_dissim_img_arr['arr_{}'.format(img_idx)]
            dissim_img_feature = dissim_img_feature[:, :max_img_num, :]
            img_idx += 1
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            e=tf.train.Example.FromString(example_str)
            try:
                article_text = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
                abstract_text = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
                url_hash = e.features.feature['url_hash'].bytes_list.value[0]
            except ValueError:
                logger.error('Failed to get article or abstract or url_hash from example')
                continue
            if len(article_text)==0: 
                logger.warning('Found an example with empty article text. Skipping it.')
            elif len(abstract_text)==0:
                logger.warning('Found an example with empty abstract text. Skipping it.')
            else:
                yield (url_hash, article_text , abstract_text , img_feature, sim_img_feature, dissim_img_feature)



def example_generator(raw_dataset,params,vocab,batch_size):
    #Example generator
    for raw_record in raw_dataset:
        url_hash = raw_record[0].numpy().decode("utf-8") 
        article=raw_record[1].numpy().decode("utf-8")
        abstract=raw_record[2].numpy().decode("utf-8")
        img_feature=raw_record[3]
        if params.mode == "train":
            sim_img_feature = tf.squeeze(raw_record[4],axis=0) #shape (10, img_embed_dim)
            dissim_img_feature=tf.squeeze(raw_record[5], axis=0)
        img_feature = tf.squeeze(img_feature, axis=0) # shape == (img_num, img_embed_dim)
        img_num = img_feature.shape[0]
        abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)]
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)
        # Process the article
        article_words = article.split()
        if len(article_words) > params.max_enc_steps:
            article_words = article_words[:params.max_enc_steps]
        enc_len = len(article_words)  # store the length after truncation but before padding
        enc_input = [vocab.word2id(w) for w in article_words] # list of word ids; OOVs are represented by the id for UNK token
        # Process the abstract
        abstract = ' '.join(abstract_sentences) # string
        abstract_words = abstract.split() # list of strings
        abs_ids = [vocab.word2id(w) for w in abstract_words] # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        dec_input, target = data.get_dec_inp_targ_seqs(abs_ids, params.max_dec_steps, start_decoding, stop_decoding)
        dec_len = len(dec_input)

        #Testing if any of the lengths is zero or not
        if img_num==0 or enc_len==0 or dec_len==0:
            continue

        if params.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            enc_input_extend_vocab, article_oovs = data.article2ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, target = data.get_dec_inp_targ_seqs(abs_ids_extend_vocab, params.max_dec_steps, start_decoding, stop_decoding)
        

        enc_mask=get_mask(enc_len,enc_len)
        dec_mask=get_mask(dec_len,params.max_dec_steps)
        if params.mode == "train" and "SIMPAD" in params.experiment:
            img_mask=get_mask(img_num,params.max_img_num)
        else:
            img_mask=get_mask(img_num,img_num)
        
        
        
        output = {
        "enc_len": enc_len,
        "enc_input": enc_input,
        "enc_input_extend_vocab": enc_input_extend_vocab,
        "enc_mask": enc_mask,
        "article_oovs": article_oovs,
        "dec_input": dec_input,
        "target": target,
        "dec_len": dec_len,
        "dec_mask":dec_mask,
        "article": article,
        "abstract": abstract,
        "abstract_sents": abstract_sentences,
        "img_feature": img_feature,
        "img_num":img_num,
        "img_mask":img_mask,
        "url_hash" : url_hash
        }

        if params.mode == "train":
            output["sim_img_feature"] = sim_img_feature
            output["dissim_img_feature"] = dissim_img_feature
            if "SIMPAD" in params.experiment and img_num < params.max_img_num:
                sim_padded_img_feature = tf.concat([img_feature,sim_img_feature[:(params.max_img_num - img_num),:]],axis = 0)
                output["img_feature"] = sim_padded_img_feature
            yield output
        else:
            for _ in range(batch_size):
                yield output
            




def batch_generator(generator,raw_dataset,params,vocab,batch_size):
    output_types_dict = {
        "enc_len": tf.int32,
        "enc_input": tf.int32,
        "enc_input_extend_vocab": tf.int32,
        "enc_mask":tf.float32,
        "article_oovs": tf.string,
        "dec_input": tf.int32,
        "target": tf.int32,
        "dec_len": tf.int32,
        "dec_mask": tf.float32,
        "article": tf.string,
        "abstract": tf.string,
        "abstract_sents": tf.string,
        "img_feature": tf.float32,
        "img_num":tf.int32,
        "img_mask":tf.float32,
        "url_hash" : tf.string
        }
    
    if params.mode == "train":
        output_types_dict["sim_img_feature"] = tf.float32
        output_types_dict["dissim_img_feature"] = tf.float32
    
    dataset = tf.data.Dataset.from_generator(
        lambda: generator(raw_dataset,params,vocab,batch_size),
        output_types = output_types_dict)
    
    img_feature_dim=params.img_embed_dim
    padded_shapes_dict = {"enc_len": [],
                            "enc_input": [None],
                            "enc_input_extend_vocab": [None],
                            "enc_mask":[None],
                            "article_oovs": [None],
                            "dec_input": [params.max_dec_steps],
                            "target": [params.max_dec_steps],
                            "dec_len": [],
                            "dec_mask":[params.max_dec_steps],
                            "article": [],
                            "abstract": [],
                            "abstract_sents": [None],
                            "img_feature":[None,img_feature_dim],
                            "img_num":[],
                            "img_mask":[None],
                            "url_hash" : []
                            }
    padding_values_dict = {"enc_len": -1,
                            "enc_input": vocab.word2id(data.PAD_TOKEN),
                            "enc_input_extend_vocab": vocab.word2id(data.PAD_TOKEN),
                            "enc_mask":0.0,
                            "article_oovs": b'',
                            "dec_input": vocab.word2id(data.PAD_TOKEN),
                            "target": vocab.word2id(data.PAD_TOKEN),
                            "dec_len": -1,
                            "dec_mask":0.0,
                            "article": b"",
                            "abstract": b"",
                            "abstract_sents": b'',
                            "img_feature":0.0,
                            "img_num":-1,
                            "img_mask":0.0,
                            "url_hash" : b""
                            }
    
    if params.mode == "train":
        padded_shapes_dict["sim_img_feature"] = [None,img_feature_dim]
        padded_shapes_dict["dissim_img_feature"] = [None,img_feature_dim]
        padding_values_dict["sim_img_feature"] = 0.0
        padding_values_dict["dissim_img_feature"] = 0.0
    
    dataset = dataset.padded_batch(batch_size, padded_shapes=(padded_shapes_dict),
                                   padding_values=padding_values_dict,
                                   drop_remainder=True)
    def update(record):
        encoder_input_dict = {"enc_input": record["enc_input"],
                            "extended_enc_input": record["enc_input_extend_vocab"],
                            "article_oovs": record["article_oovs"],
                            "enc_len": record["enc_len"],
                            "enc_mask":record["enc_mask"],
                            "article": record["article"],
                            "max_oov_len": tf.shape(record["article_oovs"])[1],
                            "img_feature":record["img_feature"],
                            "img_num":record["img_num"],
                            "img_mask":record["img_mask"],
                            "url_hash" : record["url_hash"]
                }
        decoder_input_dict ={"dec_input": record["dec_input"],
                            "dec_target": record["target"],
                            "dec_len": record["dec_len"],
                            "dec_mask":record["dec_mask"],
                            "abstract_sents" : record["abstract_sents"],
                            "abstract": record["abstract"]}
        if params.mode == "train":
            encoder_input_dict["sim_img_feature"] = record["sim_img_feature"]
            encoder_input_dict["dissim_img_feature"] = record["dissim_img_feature"]
            if "SIMPAD" in params.experiment or "TS" in params.experiment:
                #In SIMPAD all the images needs to be considered and no masking is done during attention computation
                #Since original images are not considered in "DSC_MSMO-TS" so we consider all similar/dissimilar images
                batch_max_img_num = params.max_img_num
            else:
                batch_max_img_num = tf.reduce_max(record["img_num"])
            sim_ones = tf.ones([params.batch_size,batch_max_img_num], dtype = tf.int32)
            dissim_zeros = tf.zeros([params.batch_size,batch_max_img_num],dtype=tf.int32)
            decoder_input_dict["dsc_target"] = tf.concat([sim_ones, dissim_zeros], axis = -1)
        
        return (encoder_input_dict, decoder_input_dict)
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(update, num_parallel_calls=AUTOTUNE)
    return dataset



class batcher:
    def __init__(self,data_path, img_feature_path,sim_img_feature_path, dissim_img_feature_path, vocab):
        self.data_path=data_path
        self.img_feature_path=img_feature_path
        self.sim_img_feature_path = sim_img_feature_path
        self.dissim_img_feature_path = dissim_img_feature_path
        self.vocab=vocab
    
    def get_batched_dataset(self,params,batch_size, logger):
        '''Returns Batched dataset as per batch size and other parameters'''
        if params.mode == "train":
            raw_dataset = tf.data.Dataset.from_generator(lambda: raw_record_generator(self.data_path, self.img_feature_path, logger, self.sim_img_feature_path, self.dissim_img_feature_path),
                                            output_types=(tf.string, tf.string,tf.string,tf.float32, tf.float32, tf.float32))
        else:
            raw_dataset = tf.data.Dataset.from_generator(lambda: raw_record_generator(self.data_path, self.img_feature_path),
                                            output_types=(tf.string, tf.string,tf.string,tf.float32))
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # raw_dataset= raw_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        if not params.single_pass:
            # We repeat and shuffle the dataset only during train mode
            raw_dataset=raw_dataset.shuffle(1000, reshuffle_each_iteration=True).repeat()
        dataset=batch_generator(example_generator,raw_dataset,params,self.vocab,batch_size)
        dataset=dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset
