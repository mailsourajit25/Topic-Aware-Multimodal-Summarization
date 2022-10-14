import os
import tensorflow as tf
import time
import datetime 
import pytz 
import shutil
from tqdm import tqdm
from beam_search_decoder import beam_decode
import pandas as pd
from pyrouge import Rouge155
import statistics
import pprint
from utils import find_image_precison, write_for_attnvis, write_for_rouge




def calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    """Calculate the final distribution, for the pointer-generator model
    Args:
    _enc_batch_extend_vocab : Shape (batch_size, None)
    vocab_dists: Shape (batch_size, vsize), len =  max_dec_steps - Tensorarray : The words are in the order they appear in the vocabulary file.
    attn_dists: Shape ( batch_size, attn_len) len =  max_dec_steps - Tensorarray
    p_gens: Shape ( batch_size,1) len =  max_dec_steps - Tensorarray
    batch_oov_len : Max length of oov words in that batch. Shape ()
    vocab_size : Size of vocabulary
    batch_size : batch_size
    Returns:
    final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    #print("Tracing Inside Calc_final_dist")
    #tf.print("Inside Calc_final_dist")
    #max_dec_steps=tf.shape(vocab_dists)[0]
    max_dec_steps=vocab_dists.size() # Returns size tensor-array
    # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
    vocab_dists_pgen = tf.TensorArray(tf.float32, size=max_dec_steps)
    attn_dists_pgen = tf.TensorArray(tf.float32, size=max_dec_steps)
    
    for t_step in tf.range(0,limit=max_dec_steps):
        vocab_dists_pgen = vocab_dists_pgen.write(t_step, p_gens.read(t_step) * vocab_dists.read(t_step)) # Result shape == ( batch_size, vsize) , len  = dec_steps
        attn_dists_pgen = attn_dists_pgen.write(t_step, (1 - p_gens.read(t_step)) * attn_dists.read(t_step)) # Result Shape == (batch_size, attn_length) , len  = dec_steps
    
    #   vocab_dists=p_gens * vocab_dists # Result shape == (dec_steps, batch_size, vsize)
    #   attn_dists =(1 - p_gens) * attn_dists # Result Shape == (dec_steps, batch_size, attn_length)
    
    # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
    extended_vsize = vocab_size + batch_oov_len  # the maximum (over the batch) size of the extended vocabulary
    extra_zeros = tf.zeros((batch_size, batch_oov_len)) # shape (batch_size, batch_oov_len) 
    vocab_dists_extended = tf.TensorArray(tf.float32, size=max_dec_steps)
    for t_step in tf.range(0,limit=max_dec_steps):
        vocab_dists_extended = vocab_dists_extended.write(t_step,tf.concat([vocab_dists_pgen.read(t_step),extra_zeros], axis=1))
    
    #vocab_dists_extended = tf.concat([vocab_dists, extra_zeros],-1)  # shape (max_dec_steps, batch_size, extended_vsize)

    # Project the values in the attention distributions onto the appropriate entries in the final distributions
    # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
    # This is done for each decoder timestep.
    # This is fiddly; we use tf.scatter_nd to do the projection
    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
    attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
    indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
    shape = [batch_size, extended_vsize]
    attn_dists_projected=tf.TensorArray(tf.float32, size=max_dec_steps)
    for t_step in tf.range(0,limit=max_dec_steps):
        copy_dist=tf.scatter_nd(indices, attn_dists_pgen.read(t_step), shape)
        attn_dists_projected = attn_dists_projected.write(t_step,copy_dist) # shape == (batch_size, extended_vsize) len =max_dec_steps

    # Add the vocab distributions and the copy distributions together to get the final distributions
    # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
    # final_dists  is a tensor shape (max_dec_steps, batch_size, extended_vsize) giving the final distribution for that decoder timestep
    # final_dists=vocab_dists_extended + attn_dists_projected
    final_dists=tf.TensorArray(tf.float32, size=max_dec_steps, clear_after_read= False)
    for t_step in tf.range(0,limit=max_dec_steps):
        final_dists = final_dists.write(t_step, vocab_dists_extended.read(t_step) + attn_dists_projected.read(t_step) )

    return final_dists


# ### Loss Functions

def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
        values: tensor_shape (batch_size, ). len , max_dec_steps : Tensorarray
        padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
        a scalar
    """
    #print("Tracing Inside Mask and Avg")
    #tf.print("Inside Mask and Avg")
    max_dec_steps=values.size()
    dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size
    values_per_ex=tf.TensorArray(tf.float32, size=max_dec_steps)
    for t_step in tf.range(0, limit=max_dec_steps):
        values_per_ex=values_per_ex.write(t_step,values.read(t_step) * padding_mask[:,t_step]) #shape (batch_size)
    # values_per_ex is Tensorarray with len = max_dec_steps and shape of every element is (batch_size)
    values_per_ex_normalized = tf.reduce_sum(values_per_ex.stack(),axis=0)/dec_lens # shape (batch_size); normalized value for each batch member
    avg_values_per_ex=tf.reduce_mean(values_per_ex_normalized) # overall average
    return avg_values_per_ex


def _coverage_loss(attn_dists, padding_mask):
    #print("Tracing Inside Coverage Loss")
    #tf.print("Inside Coverage Loss")
    """Calculates the coverage loss from the attention distributions.

    Args:
        attn_dists:  shape (batch_size, dec_max_steps, attn_length)
        padding_mask: shape (batch_size, dec_max_steps).

    Returns:
        coverage_loss: scalar
    """
    
    coverage=tf.zeros_like(attn_dists.read(0)) # shape (batch_size, attn_length). Initial coverage is zero.
    max_dec_steps = attn_dists.size()
    coverage_loss = tf.TensorArray(tf.float32, size=max_dec_steps)
    for t_step in tf.range(0, limit=max_dec_steps):
        covloss = tf.reduce_sum( tf.math.minimum(attn_dists.read(t_step), coverage) , axis = 1 ) #shape = batch_size
        coverage_loss = coverage_loss.write(t_step, covloss)
        coverage = coverage + attn_dists.read(t_step)
    coverage_loss = _mask_and_avg(coverage_loss, padding_mask)
    return coverage_loss


def _text_loss(dec_target,final_dists, _dec_mask ):
    #print("Tracing Inside Text Loss")
    #tf.print("Inside Text Loss")
    #dec_target shape == (batch_size, dec_steps)
    #final_dists shape== (batch_size, extended_vocab_size) len = dec_steps Tensorarray
    #dec_mask shape == (batch_size, dec_steps)
    batch_size=tf.shape(dec_target)[0]
    max_dec_steps=tf.shape(dec_target)[1]
    loss_per_step = tf.TensorArray(tf.float32, size=max_dec_steps)
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    for t_step in tf.range(0, limit=max_dec_steps):
        targets = dec_target[:, t_step] #Shape batch_size
        indices = tf.stack( [batch_nums, targets], axis=1) #shape (batch_size,2)
        gold_probs = tf.gather_nd(final_dists.read(t_step), indices) # shape = (batch_size)
        losses = -tf.math.log(gold_probs) # shape = (batch_size)
        loss_per_step = loss_per_step.write(t_step, losses)
    
    # Apply dec_padding_mask and get loss
    _loss = _mask_and_avg(loss_per_step, _dec_mask)
    return _loss

bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction = tf.keras.losses.Reduction.NONE)

def _classifier_loss(y_reals,y_preds, img_mask):
    # print("Tracing Inside Classifier Loss")
    #y_reals, y_preds shape == (batch_size , 2 * batch_max_img_num)
    #img_mask  shape == (batch_size , batch_max_img_num)
    img_mask = tf.tile(img_mask,[1,2]) #shape == (batch_size , 2 * batch_max_img_num)
    img_nums = tf.reduce_sum(img_mask, axis=1) # shape batch_size
    #Expanding dimensions so as to apply mask
    y_reals = tf.expand_dims(y_reals, axis = -1) # shape = (batch_size, 2 * batch_max_img_num , 1)
    y_preds = tf.expand_dims(y_preds, axis = -1)
    all_loss = bce_loss(y_reals, y_preds, sample_weight=img_mask) # shape (batch_size, 2 * batch_max_img_num)
    loss_per_sample_normalized = tf.reduce_sum(all_loss,axis=1)/img_nums # shape (batch_size); normalized value for each batch member
    avg_loss=tf.reduce_mean(loss_per_sample_normalized) # overall average
    return avg_loss



def train_model(model, dataset, params, ckpt, ckpt_manager):
    # test_log_dir = 'templogs2/gradient_tape/' + current_time + '/test'
    # train_summary_writer = tf.summary.create_file_writer(params.tfsummary_logdir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    optimizer = tf.keras.optimizers.Adagrad(params.lr ,
                                            initial_accumulator_value=params.adagrad_init_acc,
                                            global_clipnorm=params.max_grad_norm)
    classifier_acc_metric = tf.keras.metrics.BinaryAccuracy()

    @tf.function(input_signature=(tf.TensorSpec(shape=[params.batch_size, None], dtype=tf.int32),
                                tf.TensorSpec(shape=[params.batch_size,None], dtype=tf.int32),
                                tf.TensorSpec(shape=[params.batch_size,None], dtype=tf.float32),
                                tf.TensorSpec(shape=[params.batch_size,None,params.img_embed_dim], dtype=tf.float32),
                                tf.TensorSpec(shape=[params.batch_size,None], dtype=tf.float32),
                                tf.TensorSpec(shape=[params.batch_size,params.max_dec_steps], dtype=tf.int32),
                                tf.TensorSpec(shape=[params.batch_size,params.max_dec_steps], dtype=tf.int32),
                                tf.TensorSpec(shape=[params.batch_size,params.max_dec_steps], dtype=tf.float32),
                                tf.TensorSpec(shape=[params.batch_size,None,params.img_embed_dim], dtype=tf.float32),
                                tf.TensorSpec(shape=[params.batch_size,None,params.img_embed_dim], dtype=tf.float32),
                                tf.TensorSpec(shape=[params.batch_size,None], dtype=tf.int32),
                                tf.TensorSpec(shape=[], dtype=tf.int32),
                                tf.TensorSpec(shape=[params.batch_size, None], dtype=tf.float32),
                                tf.TensorSpec(shape=[params.batch_size, None], dtype=tf.float32)

    ))
    def train_step(enc_inp, enc_extended_inp, enc_mask, enc_img_inp, img_mask, dec_inp, dec_tar, dec_mask, sim_img_encoded, dissim_img_encoded, dsc_reals, batch_oov_len, prev_coverage, prev_coverage_img):
        #print("Tracing Inside train_step ")
        #When params.coverage is not true then these are the default values for cov_img_loss and coverage_loss
        coverage_loss = 0.0
        coverage_img_loss = 0.0
        #tf.print("Inside train_step ")
        with tf.GradientTape() as tape:
            enc_hidden, enc_outputs = model.call_encoder(enc_inp)
            enc_img_outputs=model.call_image_projector(enc_img_inp)
            dsc_preds = model.domain_similarity_classifier(enc_hidden, enc_img_outputs, sim_img_encoded, dissim_img_encoded)
            predictions, attn_dists, attn_dists_img= model(enc_outputs, enc_hidden, enc_mask, enc_extended_inp, enc_img_outputs,img_mask, dec_inp, batch_oov_len, prev_coverage, prev_coverage_img)
            text_loss = _text_loss(dec_tar,predictions, dec_mask )
            total_loss=text_loss
            if params.coverage:
                coverage_loss = _coverage_loss(attn_dists, dec_mask)
                coverage_img_loss = _coverage_loss(attn_dists_img, dec_mask)
                total_loss+= params.cov_loss_wt * coverage_loss +  params.cov_loss_wt_img * coverage_img_loss
            
            classifier_loss = _classifier_loss(dsc_reals,dsc_preds, img_mask)
            total_loss+= params.classifier_wt * classifier_loss
            
            predictions.close()
            attn_dists.close()
            attn_dists_img.close()
            
        # variables = model.encoder.trainable_variables +  model.image_projector.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables + model.pointer.trainable_variables + model.predict_dist.trainable_variables
        gradients = tape.gradient(total_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        
        #Computing classifier accuracy during training
        dsc_pred_logits = tf.math.sigmoid(dsc_preds)
        dsc_pred_logits = tf.expand_dims(dsc_pred_logits, -1)
        dsc_reals_expanded = tf.expand_dims(dsc_reals, -1)
        mask_weights = tf.tile(img_mask,[1,2])
        classifier_acc_metric.update_state(dsc_reals_expanded,dsc_pred_logits ,mask_weights)
        classifier_acc = classifier_acc_metric.result()
        classifier_acc_metric.reset_states()
        return total_loss, text_loss, coverage_loss, coverage_img_loss, classifier_loss, classifier_acc
        #End of nested function

    try:
        prev_coverage = tf.zeros([params.batch_size, params.max_enc_steps])
        prev_coverage_img = tf.zeros([params.batch_size, params.max_img_num])
        for (_, batch) in enumerate(dataset):
            # for batch in dataset.take(1):
            # for (_, (input_tensor, input_tensor_extend_vocab, target_tensor, target_tensor, max_art_oovs)) in enumerate(dataset):
            t0 = time.time()
            
            # tf.summary.trace_on(graph=True)
            
            total_loss, text_loss, coverage_loss, coverage_img_loss, classifier_loss, classifier_acc = train_step(batch[0]["enc_input"], batch[0]["extended_enc_input"],batch[0]["enc_mask"],
                                batch[0]["img_feature"],batch[0]["img_mask"],
                                batch[1]["dec_input"],batch[1]["dec_target"], batch[1]["dec_mask"],
                                batch[0]["sim_img_feature"], batch[0]["dissim_img_feature"], batch[1]["dsc_target"],
                                batch[0]["max_oov_len"], prev_coverage, prev_coverage_img)
            
            if (tf.math.is_nan(total_loss) or tf.math.is_nan(text_loss) or tf.math.is_nan(coverage_loss) or tf.math.is_nan(coverage_img_loss) or tf.math.is_nan(classifier_loss)):
                print(f"NaN value found at step {int(ckpt.step)}, Checkpoint not saved")
                break
            if (tf.math.is_inf(total_loss) or tf.math.is_inf(text_loss) or tf.math.is_inf(coverage_loss) or tf.math.is_inf(coverage_img_loss) or tf.math.is_inf(classifier_loss)):
                print(f"Inf value found at step {int(ckpt.step)}, Checkpoint not saved")
                break

            # with train_summary_writer.as_default():
            #     # tf.summary.trace_export(name="train_step",step=0)
            #     tf.summary.scalar('total_loss',total_loss.numpy(), step=int(ckpt.step))
            #     tf.summary.scalar('text_loss',text_loss.numpy(), step=int(ckpt.step))
            #     tf.summary.scalar('classifier_loss',classifier_loss.numpy(), step=int(ckpt.step))
            #     tf.summary.scalar('classifier_acc',classifier_acc.numpy(), step=int(ckpt.step))
            #     if params.coverage:
            #         tf.summary.scalar('coverage_loss',coverage_loss.numpy(), step=int(ckpt.step))
            #         tf.summary.scalar('coverage_loss_img',coverage_img_loss.numpy(), step=int(ckpt.step))
            
            # print(train_step.get_concrete_function(batch[0]["enc_input"], batch[0]["extended_enc_input"],batch[0]["enc_mask"],
            #                     batch[0]["img_feature"],batch[0]["img_mask"],
            #                     batch[1]["dec_input"],batch[1]["dec_target"], batch[1]["dec_mask"], batch[0]["max_oov_len"]).graph)
            
            template='Exp: {} ,Step {}, time {:.2f}, Total_Loss {:.4f}, Text_Loss {:.4f}, Classifier_Loss {:.4f}, Classifier_Accuracy {:.4f}'
            print(template.format(params.experiment,
                                int(ckpt.step),
                                time.time() - t0,
                                total_loss.numpy(),
                                text_loss.numpy(),
                                classifier_loss.numpy(),
                                classifier_acc.numpy()
                                ))
            if params.coverage:
                print("Coverage_Loss {:.4f}, Coverage_Img_Loss {:.4f}".format(coverage_loss.numpy(),
                                coverage_img_loss.numpy()))
            
            if int(ckpt.step) == params.max_train_iterations:
                ckpt_manager.save(checkpoint_number=int(ckpt.step))
                print("Saved checkpoint for step {}".format(int(ckpt.step)))
                break
            if int(ckpt.step) % params.checkpoint_save_steps == 0:
                ckpt_manager.save(checkpoint_number=int(ckpt.step))
                print("Saved checkpoint for step {}".format(int(ckpt.step)))
            ckpt.step.assign_add(1)


    except KeyboardInterrupt:
        ckpt_manager.save(int(ckpt.step))
        print("Saved checkpoint for step {}".format(int(ckpt.step)))
    
    print("Done")



def eval_model(model, batch, params, prev_coverage, prev_coverage_img):

    enc_hidden, enc_outputs = model.call_encoder(batch[0]["enc_input"])
    enc_img_outputs=model.call_image_projector(batch[0]["img_feature"])
    predictions, attn_dists, attn_dists_img= model(enc_outputs, enc_hidden, batch[0]["enc_mask"], batch[0]["extended_enc_input"], enc_img_outputs,batch[0]["img_mask"],batch[1]["dec_input"], batch[0]["max_oov_len"], prev_coverage, prev_coverage_img)
    text_loss = _text_loss(batch[1]["dec_target"],predictions, batch[1]["dec_mask"] )
    if params.coverage:
        coverage_loss = _coverage_loss(attn_dists, batch[1]["dec_mask"])
        coverage_img_loss = _coverage_loss(attn_dists_img, batch[1]["dec_mask"])
    predictions.close()
    attn_dists.close()
    attn_dists_img.close()
    if params.coverage:
        return [text_loss, coverage_loss, coverage_img_loss]
    else:
        return [text_loss]


def test_and_save(params, logger, batch_iterator, model, vocab, latest_checkpoint):
    test_df = pd.read_csv(params.test_csv_path)
    test_df.set_index("URL_Hash", drop=False, inplace=True)
    image_precision = []
    image_precision2 = []
    domain_count = {}
    domain_image_precision = {}
    domain_image_precision2 = {}
    
    attn_vis_result_dir = os.path.join(params.test_save_dir,"attn_vis")
    if not os.path.exists(attn_vis_result_dir):
        os.makedirs(attn_vis_result_dir)
    rouge_ref_dir = os.path.join(params.test_save_dir,"rouge_dir","rouge_ref_dir")
    rouge_dec_dir = os.path.join(params.test_save_dir,"rouge_dir","rouge_dec_dir")
    
    logger.info(f"Testing for experiment - {params.experiment}")
    #For mapping hash and index of the saved summaries
    hash_f = open(os.path.join(attn_vis_result_dir,"hash_idx.txt"),"w")
    
    with tqdm(total=params.num_to_test, position=0, leave=True) as pbar:
        for i in range(params.num_to_test):
            batch = next(batch_iterator, None)
            
            # Extra code - remove it later
            # if batch[0]["url_hash"].numpy()[0].decode() not in test_df["URL_Hash"].values:
            #     pbar.update(1)
            #     continue
            
            if batch==None:
                #End of generator
                break
            trial = beam_decode(model, batch, vocab, params)
            hash_f.write(f"{i},{trial.url_hash}\n")
            sample_domain = test_df.loc[trial.url_hash]["Domain"]
            coverage_img_scores = list(map(lambda x: str(x.numpy()),trial.coverage_img))
            ip_score, ip_score2, ref_imgs, pred_imgs = find_image_precison(trial.url_hash, coverage_img_scores, test_df, trial.img_num)
            image_precision.append(ip_score)
            image_precision2.append(ip_score2)
            if sample_domain not in domain_count:
                domain_count[sample_domain] = 0
                domain_image_precision[sample_domain] = [ip_score]
                domain_image_precision2[sample_domain] = [ip_score2]
            else:
                domain_count[sample_domain]+=1
                domain_image_precision[sample_domain].append(ip_score)
                domain_image_precision2[sample_domain].append(ip_score2)

            if i%params.test_save_steps == 0:
                attn_vis_result_file = os.path.join(attn_vis_result_dir,f"article_{i}.txt")
                # attn_vis_result_file = os.path.join(attn_vis_result_dir,f"{trial.url_hash}.txt")
                with open(attn_vis_result_file, "w") as f:
                    f.write("Article:\n")
                    f.write(trial.real_article)
                    f.write("\n\nPredicted abstract:\n")
                    f.write(trial.predicted_abstract)
                    f.write("\n\nReal abstract:\n")
                    f.write(trial.real_abstract)
                    f.write("\n\nCoverage img scores:\n")
                    f.write(",".join(coverage_img_scores))
                    f.write("\n\nURL_Hash:\n")
                    f.write(trial.url_hash)
                    f.write("\n\nImage Precision Score:\n")
                    f.write(str(round(ip_score*100,2)))
                    f.write("\n\nImage Precision Score - 2:\n")
                    f.write(str(round(ip_score2*100,2)))
                    f.write("\n\nReference image indices:\n")
                    f.write(ref_imgs)
                    f.write("\n\nPredicted image indices:\n")
                    f.write(pred_imgs)
                    f.write("\n\nPredicted abstract with unks:\n")
                    f.write(trial.pred_abstract_with_unks)
                    f.write("\n\nReal abstract with unks:\n")
                    f.write(trial.abstract_with_unks)
                    logger.info(f"{i}th Test Article written ")
                    write_for_attnvis(params,trial,i,params.test_img_path, ref_imgs, pred_imgs) # write info to .json file for visualization tool
            write_for_rouge(trial.real_abstract_sents, trial.decoded_words, i,domain_count[sample_domain], sample_domain, rouge_ref_dir, rouge_dec_dir)
            pbar.update(1)
    hash_f.close()
    result_folder = os.path.join(params.test_save_dir,"results")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    r = Rouge155()
    with open(os.path.join(result_folder,f"final_result.txt"),"w") as output_file:
        output_file.write(f"\nInferenced from checkpoint: {latest_checkpoint}\n")
        output_file.write(f"Time : {datetime.datetime.now(pytz.timezone('Asia/Kolkata'))} ")
        output_file.write(f"\nNo. of test articles: {params.num_to_test}\n")
        output_file.write((f"\nParams: {params}\n"))
        r.system_dir = rouge_dec_dir
        r.model_dir = rouge_ref_dir
        r.system_filename_pattern = '(\d+)_decoded.txt'
        r.model_filename_pattern = '#ID#_reference.txt'
        scores = r.convert_and_evaluate()
        print(f"\n\n{params.experiment} - Rouge Scores")
        pprint.pprint(scores)
        output_file.write(f"\n\n{params.experiment} - Rouge Scores")
        output_file.write(pprint.pformat(scores))
        print(f"\n\n{params.experiment} -Image Precision Scores -1")
        print(round(statistics.mean(image_precision)*100,2))
        output_file.write(f"\n\n{params.experiment} -Image Precision Scores -1")
        output_file.write(f"  {round(statistics.mean(image_precision)*100,2)}")
        print(f"\n\n{params.experiment} -Image Precision Scores -2")
        print(round(statistics.mean(image_precision2)*100,2))
        output_file.write(f"\n\n{params.experiment} -Image Precision Scores -2")
        output_file.write(f"  {round(statistics.mean(image_precision2)*100,2)}")
        output_file.write('\n'+'=' * 40+'\n')
        output_file.write("Domain wise scores")
        output_file.write('\n'+'=' * 40+'\n')
        for _domain in domain_count:
            r_domain = Rouge155()
            domain_rouge_ref_dir = os.path.join(os.path.dirname(rouge_ref_dir),f"{_domain}_ref")
            domain_rouge_dec_dir = os.path.join(os.path.dirname(rouge_dec_dir),f"{_domain}_dec")
            r_domain.system_dir = domain_rouge_dec_dir
            r_domain.model_dir = domain_rouge_ref_dir
            r_domain.system_filename_pattern = '(\d+)_decoded.txt'
            r_domain.model_filename_pattern = '#ID#_reference.txt'
            domain_scores = r_domain.convert_and_evaluate()
            output_file.write(f"\n\n{_domain} - {params.experiment} - Rouge Scores ")
            output_file.write(pprint.pformat(domain_scores))
            output_file.write(f"\n\n{_domain} - {params.experiment} -Image Precision Scores -1")
            output_file.write(f"  {round(statistics.mean(domain_image_precision[_domain])*100,2)}")
            output_file.write(f"\n\n{_domain} - {params.experiment} -Image Precision Scores -2")
            output_file.write(f"  {round(statistics.mean(domain_image_precision2[_domain])*100,2)}")
            output_file.write('\n'+'=' * 40+'\n')
            output_file.write('\n'+'=' * 40+'\n')

    # shutil.rmtree(os.path.join(params.test_save_dir,"rouge_dir"))
    print("Done")
    


    

    
    
    


