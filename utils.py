import os
import tensorflow as tf
import numpy as np
import glob
import shutil
import json



### Checkpoint Restore functions
def get_latest_model(ckpt_manager):
    '''Returns the latest checkpoint that does not contain NaN'''
    checkpoint_names = ckpt_manager.checkpoints
    best_ckpt=""
    for ckpt_name in reversed(checkpoint_names):
        if not hasNaN(ckpt_name):
            best_ckpt = ckpt_name
            break
        else:
            print("NaN checkpoint found")
    
    return best_ckpt

def hasNaN(checkpoint_name):
    '''Checks if given checkpoint has any nan stored in it'''
    reader = tf.train.load_checkpoint(checkpoint_name)
    # finite = []
    all_infnan = []
    some_infnan = []
    var_to_shape_map = reader.get_variable_to_shape_map()
    # dtype_from_key = reader.get_variable_to_dtype_map()
    for key in sorted(var_to_shape_map.keys()):
        tensor = reader.get_tensor(key)
        if "CHECKPOINTABLE_OBJECT_GRAPH" in key:
            break
        if np.all(np.isfinite(tensor)):
            # finite.append(key)
            continue
        else:
            if not np.any(np.isfinite(tensor)):
                all_infnan.append(key)
            else:
                some_infnan.append(key)

    if not all_infnan and not some_infnan:
        # print("CHECK PASSED: checkpoint contains no inf/NaN values")
        return False
    else:
        # print("CHECK FAILED: checkpoint contains some inf/NaN values")
        return True



def find_image_precison(url_hash, cov_img_scores, test_df, img_num):
    ref_imgs = list(map(int,test_df.loc[url_hash]["Ref_Img"].split(",")))
    ref_imgs = np.array(ref_imgs)
    num_ref_imgs = len(ref_imgs)
    cov_img_scores= np.array(cov_img_scores)
    cov_img_scores = cov_img_scores[:img_num]
    pred_imgs = cov_img_scores.argsort()[::-1][:num_ref_imgs]
    pred_imgs = pred_imgs + 1 #since argsort returns zero based indices
    matched_imgs = np.intersect1d(pred_imgs, ref_imgs)
    ip_score = matched_imgs.shape[0]/ref_imgs.shape[0]
    #ip_score 2 calculation
    pred_imgs2 = cov_img_scores.argsort()[::-1][:num_ref_imgs]
    pred_imgs2 = pred_imgs2 + 1 #since argsort returns zero based indices
    matched_imgs = np.intersect1d(pred_imgs2, ref_imgs)
    ip_score2 = matched_imgs.shape[0]/img_num
    pred_imgs_str = np.char.mod('%d', pred_imgs)
    pred_imgs_str = ",".join(pred_imgs_str)
    return ip_score,ip_score2, test_df.loc[url_hash]["Ref_Img"], pred_imgs_str

def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

def write_for_attnvis(params, best_hyp, index, parent_img_dir, ref_imgs, pred_imgs):
    article_lst = best_hyp.article_with_unks.split() # list of words
    decoded_lst = best_hyp.predicted_abstract.split() # list of decoded words
    attn_dists = list(map(lambda x: tf.squeeze(x).numpy().tolist(),best_hyp.attn_dists))
    attn_dists_img = list(map(lambda x: tf.squeeze(x).numpy().tolist(),best_hyp.attn_dists_img))
    p_gens = list(map(float,best_hyp.p_gens))
    to_write = {
        'article_lst': [make_html_safe(t) for t in article_lst],
        'decoded_lst': [make_html_safe(t) for t in decoded_lst],
        'abstract_str': make_html_safe(best_hyp.abstract_with_unks),
        'attn_dists': attn_dists,
        'attn_dists_img': attn_dists_img,
        'url_hash': best_hyp.url_hash,
        'parent_img_dir': parent_img_dir,
        'p_gens': p_gens,
        'ref_imgs': ref_imgs,
        'pred_imgs': pred_imgs,
        'experiment': params.experiment,
        'img_num': int(best_hyp.img_num)
    }
    attn_vis_result_dir = os.path.join(params.test_save_dir,"attn_vis")
    if not os.path.exists(attn_vis_result_dir):
        os.makedirs(attn_vis_result_dir)
    output_fname = os.path.join(attn_vis_result_dir, f'attn_vis_data_{index}.json')
    with open(output_fname, 'w') as output_file:
        json.dump(to_write, output_file)
    print(f'Wrote visualization data to {output_fname}')

def write_for_rouge(reference_sents, decoded_words, ex_index, domain_ex_index, domain,_rouge_ref_dir, _rouge_dec_dir):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """
    if not os.path.exists(_rouge_ref_dir):
        os.makedirs(_rouge_ref_dir)
    if not os.path.exists(_rouge_dec_dir):
        os.makedirs(_rouge_dec_dir)
    domain_rouge_ref_dir = os.path.join(os.path.dirname(_rouge_ref_dir),f"{domain}_ref")
    domain_rouge_dec_dir = os.path.join(os.path.dirname(_rouge_dec_dir),f"{domain}_dec")
    if not os.path.exists(domain_rouge_ref_dir):
        os.makedirs(domain_rouge_ref_dir)
    if not os.path.exists(domain_rouge_dec_dir):
        os.makedirs(domain_rouge_dec_dir)
    # First, divide decoded output into sentences
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError: # there is text remaining that doesn't end in "."
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx+1] # sentence up to and including the period
        decoded_words = decoded_words[fst_period_idx+1:] # everything else
        decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w.decode()) for w in reference_sents]

    # Write to file
    ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % (ex_index))
    decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % (ex_index))
    
    dom_ref_file = os.path.join(domain_rouge_ref_dir, "%06d_reference.txt" % (domain_ex_index))
    dom_decoded_file = os.path.join(domain_rouge_dec_dir, "%06d_decoded.txt" % (domain_ex_index))

    with open(ref_file, "w") as f1, open(dom_ref_file, "w") as f2:
      for idx,sent in enumerate(reference_sents):
        f1.write(sent) if idx==len(reference_sents)-1 else f1.write(sent+"\n")
        f2.write(sent) if idx==len(reference_sents)-1 else f2.write(sent+"\n")
    with open(decoded_file, "w") as f1, open(dom_decoded_file, "w") as f2:
      for idx,sent in enumerate(decoded_sents):
        f1.write(sent) if idx==len(decoded_sents)-1 else f1.write(sent+"\n")
        f2.write(sent) if idx==len(decoded_sents)-1 else f2.write(sent+"\n")


def calc_running_avg_loss(loss, running_avg_loss, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
        loss: loss on the most recent eval step
        running_avg_loss: running_avg_loss so far
        decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
        running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    return running_avg_loss


def get_embedding(max_features, embed_size,  vocab, embd_model_path, params):
    """
    use this function only if you want to use custom pre-trained embeddings
    give vocab ,max_features,embed_size,return embedding
    Args: vocab ,max_features,embed_size
    output:embedding
    """
    # load model
    # abspath = os.path.abspath("./")
    # model_path = os.path.join(abspath, embd_model_path)
    
    # w2v_model = KeyedVectors.load_word2vec_format(embd_model_path, binary=True)
    
    # init embedding
    embedding_initializer=tf.keras.initializers.TruncatedNormal(stddev=params.trunc_norm_init_std)
    embedding = embedding_initializer(shape=(max_features, embed_size)) 
    
    # gen embedding
    # for word, id in vocab._word_to_id.items():
    #     if id <= max_features:
    #         try:
    #             embedding[id] = w2v_model[word]
    #         except KeyError:
    #             continue

    return embedding
