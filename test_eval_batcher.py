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


def raw_record_generator(data_path, img_feature_path,logger, max_img_num=10):

    filelist = glob.glob(data_path) # get the list of datafiles
    feature_list = glob.glob(img_feature_path)
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    assert feature_list, ('Error: Empty feature_list at %s' % img_feature_path)

    filelist = sorted(filelist)
    feature_list = sorted(feature_list)
    text_img_pairs = list(zip(filelist, feature_list))

    

    for text_f, img_f in text_img_pairs:
      reader = open(text_f, 'rb')
      r = np.load(img_f)
      img_idx = 0
      img_num = len(list(r))
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        if img_idx == img_num: break
        img_feature = r['arr_{}'.format(img_idx)]
        img_feature = img_feature[:, :max_img_num, :]
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
            yield (url_hash, article_text , abstract_text , img_feature)



def example_generator(raw_dataset,params,vocab,batch_size):
    for raw_record in raw_dataset:
        url_hash = raw_record[0].numpy().decode("utf-8") 
        article=raw_record[1].numpy().decode("utf-8")
        abstract=raw_record[2].numpy().decode("utf-8")
        img_feature=raw_record[3]
        img_feature = tf.squeeze(img_feature, axis=0)
        img_num = img_feature.shape[0]
        img_dim = img_feature.shape[1]
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
        "img_dim":img_dim,
        "img_mask":img_mask,
        "url_hash" : url_hash
        }
        if params.mode == "test" or params.mode == "eval":
            for _ in range(batch_size):
                    yield output
        else:
            yield output



def batch_generator(generator,raw_dataset,params,vocab,batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator(raw_dataset,params,vocab,batch_size),
        output_types = {
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
        "img_dim":tf.int32,
        "img_mask":tf.float32,
        "url_hash" : tf.string
        })

    dataset = dataset.padded_batch(batch_size, padded_shapes=({"enc_len": [],
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
                                                               "img_feature":[None,params.img_embed_dim],
                                                               "img_num":[],
                                                               "img_dim":[],
                                                               "img_mask":[None],
                                                               "url_hash" : []
                                                              }),
                                   padding_values={"enc_len": -1,
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
                                                   "img_dim":-1,
                                                   "img_mask":0.0,
                                                   "url_hash" : b""
                                                  },
                                   drop_remainder=True)
    def update(record):
        return ({"enc_input": record["enc_input"],
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
                },

                {"dec_input": record["dec_input"],
                 "dec_target": record["target"],
                 "dec_len": record["dec_len"],
                 "dec_mask":record["dec_mask"],
                 "abstract_sents" : record["abstract_sents"],
                 "abstract": record["abstract"]})
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(update, num_parallel_calls=AUTOTUNE)
    return dataset
        

class batcher:
    def __init__(self,data_path, img_feature_path,vocab):
        self.data_path=data_path
        self.img_feature_path=img_feature_path
        self.vocab=vocab
    
    def get_batched_dataset(self,params,batch_size, logger):
        '''Returns Batched dataset as per batch size and other parameters'''
        raw_dataset = tf.data.Dataset.from_generator(lambda: raw_record_generator(self.data_path, self.img_feature_path, logger),
                                        output_types=(tf.string, tf.string,tf.string,tf.float32))
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # raw_dataset= raw_dataset.cache().prefetch(buffer_size=AUTOTUNE)
        if not params.single_pass:
            # We repeat and shuffle the dataset only during train mode
            raw_dataset=raw_dataset.shuffle(1000, reshuffle_each_iteration=True).repeat()
        dataset=batch_generator(example_generator,raw_dataset,params,self.vocab,batch_size)
        dataset=dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset
