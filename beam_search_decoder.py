import data
import tensorflow as tf
import numpy as np



class Hypothesis:
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""
    def __init__(self,tokens, log_probs, state, attn_dists, attn_dists_img, p_gens, coverage, coverage_img):
        """Hypothesis constructor."""
        self.tokens = tokens # list of all the tokens from time 0 to the current time step t
        self.log_probs = log_probs # list of the log probabilities of the tokens of the tokens
        self.state = state # decoder state after the last token decoding
        self.attn_dists = attn_dists # attention dists of all the tokens
        self.attn_dists_img = attn_dists_img # attention dists of all the images
        self.p_gens = p_gens  # generation probability of all the tokens
        self.coverage = coverage # Text coverage
        self.coverage_img = coverage_img # Image coverage
        self.decoded_words = [] #list of decoded words
        self.predicted_abstract = ""
        self.real_article = ""
        self.real_abstract = ""
        self.real_abstract_sents = []
        self.article_with_unks = ""
        self.abstract_with_unks = ""
        self.pred_abstract_with_unks = ""
        self.url_hash = ""
        self.img_num = 0

        

    def extend(self, token, log_prob, state, attn_dist, attn_dist_img, p_gen, coverage, coverage_img):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search."""
        return Hypothesis(tokens = self.tokens + [token], # we add the decoded token
                        log_probs = self.log_probs + [log_prob], # we add the log prob of the decoded token
                        state = state, # we update the state
                        attn_dists = self.attn_dists + [attn_dist], # we  add the attention dist of the decoded token
                        attn_dists_img = self.attn_dists_img + [attn_dist_img],
                        p_gens = self.p_gens + [p_gen], # we add the p_gen 
                        coverage = coverage,
                        coverage_img = coverage_img)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)


def decode_onestep(model, batch, enc_outputs, enc_img_outputs, dec_state, dec_input, prev_coverage, prev_coverage_img, params):
    """
        Method to decode the output step by step (used for beamSearch decoding)
        Returns: A dictionary of the results of all the ops computations (see below for more details)
    """
    # dictionary of all the ops that will be computed
    # final_dists, dec_hidden, context_vector, attentions, p_gens, _ = model(enc_outputs, dec_state,batch[0]["enc_input"], batch[0]["extended_enc_input"], dec_input, batch[0]["max_oov_len"])
    final_dists, attentions , attentions_img, p_gens, dec_hidden, context_vector, coverage, coverage_img= model(enc_outputs, dec_state, batch[0]["enc_mask"],
                                                                     batch[0]["extended_enc_input"], enc_img_outputs,batch[0]["img_mask"], 
                                                                     dec_input, batch[0]["max_oov_len"],prev_coverage, prev_coverage_img)
    top_k_probs, top_k_ids = tf.math.top_k(tf.squeeze(final_dists), k = params.beam_size*2)
    top_k_log_probs = tf.math.log(top_k_probs)
    results = { "last_context_vector" : context_vector,
                "dec_state" : dec_hidden,
                "attentions" :attentions,
                "attentions_img":attentions_img,
                "top_k_ids" : top_k_ids,
                "top_k_log_probs" : top_k_log_probs,
                "p_gen" : p_gens,
                "coverage" : coverage,
                "coverage_img" : coverage_img
                }
    return results


def beam_decode(model, batch, vocab, params):
    # We run the encoder once and then we use the results to decode each time step token

    state, enc_outputs = model.call_encoder(batch[0]["enc_input"]) #state is the last hidden state [hidden state, cell state]
    enc_img_outputs=model.call_image_projector(batch[0]["img_feature"])
    #Here a batch contains the same example repeated "batch size" times hence initial cell state(state[0]) and hidden state (state[1]) are same for all the examples in the batch
    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)], # we initalize all the beam_size hypothesises with the token start
                      log_probs=[0.0], # Initial log prob = 0
                      state=[state[0][0], state[1][0]], #initial dec_state (we will use only the first dec_state because they're initially the same)
                      attn_dists=[],
                      attn_dists_img=[],
                      p_gens=[],
                      coverage=np.zeros([tf.shape(batch[0]["enc_input"])[1]], dtype= np.float32), # zero vector of length attention_length,
                      coverage_img=np.zeros([tf.shape(batch[0]["img_feature"])[1]], dtype= np.float32) # we init the coverage vector to zero
                      ) for _ in range(params.batch_size)]


    results = [] # list to hold the top beam_size hypothesises
    steps=0 # initial step

    while steps < params.max_dec_steps and len(results) < params.beam_size : 
        latest_tokens = [h.latest_token for h in hyps] # latest token id's produced by each hypothesis
        latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens] # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
        hidden_states = [h.state[0] for h in hyps] #Collecting list of lstm decoder hidden states
        cell_states = [h.state[1] for h in hyps] # #Collecting list of lstm decoder cell states of the hypotheses
        prev_coverage = [h.coverage for h in hyps] # list of coverage vectors (or None)
        prev_coverage_img = [h.coverage_img for h in hyps]

        # we decode the top likely 2 x beam_size tokens tokens at time step t for each hypothesis
        returns = decode_onestep( model, batch, enc_outputs, enc_img_outputs,[tf.stack(hidden_states, axis=0),tf.stack(cell_states, axis=0)], tf.expand_dims(latest_tokens, axis=1),tf.stack(prev_coverage,0), tf.stack(prev_coverage_img,0), params)
        topk_ids, topk_log_probs, new_states, attn_dists,attn_dists_img, p_gens, new_coverage, new_coverage_img =  returns['top_k_ids'],\
                                                                                                                    returns['top_k_log_probs'],\
                                                                                                                    returns['dec_state'],\
                                                                                                                    returns['attentions'],\
                                                                                                                    returns['attentions_img'],\
                                                                                                                    np.squeeze(returns["p_gen"]),\
                                                                                                                    returns["coverage"],\
                                                                                                                    returns["coverage_img"]
        all_hyps = []
        num_orig_hyps = 1 if steps ==0 else len(hyps)
        for i in range(num_orig_hyps):
            h, new_state, attn_dist, attn_dist_img, p_gen, new_coverage_i, new_coverage_img_i =  hyps[i],\
                                                                                                  [new_states[0][i], new_states[1][i]],\
                                                                                                  attn_dists[i],\
                                                                                                  attn_dists_img[i],\
                                                                                                  p_gens[i],\
                                                                                                  new_coverage[i],\
                                                                                                  new_coverage_img[i]

            for j in range(params.beam_size*2):
            # we extend each hypothesis with each of the top k tokens (this gives 2 x beam_size new hypothesises for each of the beam_size old hypothesises)
                new_hyp = h.extend(token=topk_ids[i,j].numpy(),
                                log_prob=topk_log_probs[i,j].numpy(),
                                state = new_state,
                                attn_dist=attn_dist,
                                attn_dist_img = attn_dist_img,
                                p_gen=p_gen,
                                coverage=new_coverage_i,
                                coverage_img=new_coverage_img_i
                                )
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        hyps = [] # will contain hypotheses for the next step
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
        for h in sorted_hyps: # in order of most likely h
            if h.latest_token == vocab.word2id(data.STOP_DECODING): # if stop token is reached...
                # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                if steps >= params.min_dec_steps:
                  results.append(h)
            else: # hasn't reached stop token, so continue to extend this hypothesis
                hyps.append(h)
            # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
            if len(hyps) == params.beam_size or len(results) == params.beam_size:
                break

        steps += 1
        #End of loop
    
    # At this point, either we've got beam_size results, or we've reached maximum decoder steps
    if len(results)==0:
        results=hyps

    # At the end of the loop we return the most likely hypothesis, which holds the most likely ouput sequence, given the input fed to the model
    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    best_hyp = hyps_sorted[0]

    # Extract the output ids from the hypothesis and convert back to words
    output_ids = [int(t) for t in best_hyp.tokens[1:]]
    decoded_words = data.outputids2words(output_ids, vocab, ( batch[0]["article_oovs"][0]))
    # Remove the [STOP] token from decoded_words, if necessary
    try:
        fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
        decoded_words = decoded_words[:fst_stop_idx]
    except ValueError:
        decoded_words = decoded_words
    best_hyp.decoded_words = decoded_words
    best_hyp.predicted_abstract = ' '.join(decoded_words) # single string
    best_hyp.real_abstract = batch[1]["abstract"].numpy()[0].decode()
    best_hyp.real_abstract_sents = batch[1]["abstract_sents"].numpy()[0]
    best_hyp.real_article = batch[0]["article"].numpy()[0].decode()
    best_hyp.url_hash = batch[0]["url_hash"].numpy()[0].decode()
    best_hyp.img_num = batch[0]["img_num"].numpy()[0]
    best_hyp.article_with_unks = data.show_art_oovs(best_hyp.real_article, vocab) # string
    best_hyp.abstract_with_unks = data.show_abs_oovs(best_hyp.real_abstract, vocab,  batch[0]["article_oovs"][0]) # string
    best_hyp.pred_abstract_with_unks = data.show_abs_oovs(best_hyp.predicted_abstract, vocab,  batch[0]["article_oovs"][0]) # string
    return best_hyp