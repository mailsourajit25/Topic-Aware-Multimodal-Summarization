import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional,Dense,Embedding, Layer, Concatenate, Conv2D
from train_test_eval import calc_final_dist


# ### Model Layers

class Encoder(Layer):
    def __init__(self, enc_units,dec_units, embedding_layer, batch_sz ,rand_unif_init_mag, trunc_norm_init_std):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.dec_units=dec_units
        self.trunc_norm_init=tf.keras.initializers.TruncatedNormal(stddev=trunc_norm_init_std)
        self.rand_unif_init = tf.keras.initializers.RandomUniform(-rand_unif_init_mag, rand_unif_init_mag, seed=123)
        # embeddings_initializer=tf.keras.initializers.Constant(embeddings_matrix),
        self.embedding=embedding_layer
        self.bilstm = Bidirectional(LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer = 'glorot_uniform',
                                       name="encoder-lstm"))
        
        self.linear1=Dense(self.dec_units, name="enc-lin1", kernel_initializer=self.trunc_norm_init, bias_initializer=self.trunc_norm_init)
        self.linear2=Dense(self.dec_units, name="enc-lin2", kernel_initializer=self.trunc_norm_init, bias_initializer=self.trunc_norm_init)
        

    def call(self, enc_inp):
        #print("Tracing Inside Encoder..")
        #tf.print("Inside Encoder..")
        #enc_inp shape : (batch_size, seq_length)
        embedded_inp = self.embedding(enc_inp)
        #embedded_inp after embedding shape : (batch_size, seq_length, embed_dim)
        seq_output, last_f_h, last_f_c,last_b_h, last_b_c = self.bilstm(embedded_inp) #For LSTM
        # seq_output, last_f_h, last_b_h = self.bilstm(embedded_inp)
        
        #seq_output shape : (batch_size, seq_length, 2 * enc_units ) 

        # last_f_h, last_f_c shape : (batch_size, enc_units ) - Forward LSTM
        # last_b_h, last_b_c shape : (batch_size, enc_units ) - Backward LSTM
        last_hidden=tf.nn.relu(self.linear1(tf.concat([last_f_h,last_b_h], axis = -1)))
        last_cell_state=tf.nn.relu(self.linear2(tf.concat([last_f_c,last_b_c], axis=-1))) #For LSTM
        
        #last_hidden shape : (batch_size, enc_units)
        #last_cell_state shape : (batch_size, enc_units)
        #final encoder states to be used for initializing decoder
        # encoder_states=last_hidden
        encoder_states=[last_hidden, last_cell_state] #For LSTM
        
        #encoder_states shape== (batch_size, enc_units)
        return seq_output, encoder_states



class ImageProjection(Layer):
    '''Projects embedded images to encoded text dimension'''
    def __init__(self,img_embed_dim,output_dim, trunc_norm_init_std):
        super(ImageProjection,self).__init__()
        #Here output dim = 2* enc_units because it is output dim of bidirectional LSTM
        self.img_embed_dim=img_embed_dim
        self.output_dim=output_dim
        self.trunc_norm_init=tf.keras.initializers.TruncatedNormal(stddev=trunc_norm_init_std)
        self.linear1=Dense(self.img_embed_dim,use_bias=True,name="imgproj-linear1", kernel_initializer=self.trunc_norm_init, bias_initializer=self.trunc_norm_init)
        self.linear2=Dense(self.output_dim,use_bias=True, name="imgproj-linear2", kernel_initializer=self.trunc_norm_init, bias_initializer=self.trunc_norm_init)
    
    def call(self,enc_img_inp):
        #print("Tracing Inside Image Projection Layer..")
        #tf.print("Inside Image Projection Layer..")
        #Input shape : (batch_size, max_img_num, img_embed_dim)
        batch_size=tf.shape(enc_img_inp)[0]
        enc_img_inp=tf.reshape(enc_img_inp, [-1, self.img_embed_dim])
        #Now shape = (batch_size * max_img_num, img_embed_dim)
        
        enc_img_inp=self.linear1(enc_img_inp)
        
        #Now shape = (batch_size * max_img_num, img_embed_dim)
        enc_img_inp=self.linear2(enc_img_inp)
        #Now shape of projected image vector = (batch_size * max_img_num, output_dim)
        enc_img_inp=tf.reshape(enc_img_inp,[batch_size,-1,self.output_dim])
        #Now shape of projected vector is : (batch_size, max_img_num, output_dim)
        return enc_img_inp


class MultimodalAttention(Layer):
    '''Calculates Multimodal attention, coverage and context vectors'''
    def __init__(self, attn_units, coverage, mode, trunc_norm_init_std):
        super(MultimodalAttention, self).__init__()
        self.attn_units=attn_units
        self.mode = mode 
        self.coverage = coverage
        self.trunc_norm_init=tf.keras.initializers.TruncatedNormal(stddev=trunc_norm_init_std)
        self.linear1 = Dense(self.attn_units, name="attn-linear1", use_bias=True)
        self.linear2 = Dense(self.attn_units, name="attn-linear2", use_bias=True)
        self.linear3 = Dense(self.attn_units, name="attn-linear3", use_bias=True)
        self.linear4 = Dense(self.attn_units, name="attn-linear4", use_bias=True)
        self.linearA = Dense(self.attn_units, name="attn-linearA", use_bias=False)
        self.linearB = Dense(self.attn_units, name="attn-linearB", use_bias=False)
        self.v1 = self.add_weight(name="attn-v1",shape=(self.attn_units,), trainable=True)
        self.v2 = self.add_weight(name="attn-v2",shape=(self.attn_units,), trainable=True)
        self.v3 = self.add_weight(name="attn-v3",shape=(self.attn_units,), trainable=True)
        self.v4 = self.add_weight(name="attn-v4",shape=(self.attn_units,), trainable=True)
        self.conv_w_c=self.add_weight(name="conv-w-c",shape=(1,1,1, self.attn_units), trainable=True if self.coverage else False)
        self.conv_w_c_img=self.add_weight(name="conv-w-c-img",shape=(1,1,1, self.attn_units), trainable=True if self.coverage else False)
        
        

    def masked_attention_text(self,e, mask):
        #shape of e == (batch_size, attn_length)
        #shape of mask == (batch_size, attn_length)
        attn_dist= tf.nn.softmax(e)
        attn_dist *= mask
        masked_sum = tf.reduce_sum(attn_dist, axis=1)
        return attn_dist / tf.reshape(masked_sum, [-1, 1])
    
    def masked_attention_img(self,e, mask):
        attn_dist= tf.nn.softmax(e)
        if self.mode == "train":
            #For SIMPAD model we do not apply mask as we have padded using similar images
            return attn_dist
        else:
            attn_dist *= mask
            masked_sum = tf.reduce_sum(attn_dist, axis=1)
            normalized_attn_dist = attn_dist / tf.reshape(masked_sum, [-1, 1])
            return normalized_attn_dist
    
    def call(self, coverage_set, dec_hidden, enc_outputs, enc_features, enc_mask, coverage,enc_img_outputs, enc_img_features, img_mask, coverage_img):
        #print("Tracing Inside Attention Layer..")
        #tf.print("Inside Attention Layer..")
        dec_hidden=tf.concat([dec_hidden[0], dec_hidden[1]],axis=-1) #For LSTM
        #Shape of dec_hidden after concatenating cell and hidden state is (batch_size, 2 * dec_units)

        batch_size=tf.shape(enc_features)[0]
        decoder_features=self.linear1(dec_hidden)
        decoder_img_features=self.linear2(dec_hidden)
        #shape of both is now (batch_size,attn_units)
        decoder_features=tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) 
        decoder_img_features=tf.expand_dims(tf.expand_dims(decoder_img_features, 1), 1)
        # Both of them is now reshaped to (batch_size, 1, 1, attn_units)

        if self.coverage and coverage_set : #if coverage holds some value - non first step
            coverage_features=tf.nn.conv2d(coverage, self.conv_w_c, [1, 1, 1, 1], "SAME")
            coverage_features_img=tf.nn.conv2d(coverage_img, self.conv_w_c_img, [1, 1, 1, 1], "SAME")
            #Coverage input shape : (batch_size, attn_length, 1, 1) ->Above Output shape (batch_size, attn_length, 1 , attn_units) 
            # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
            e = tf.reduce_sum(self.v1*tf.nn.tanh(enc_features + decoder_features + coverage_features), [2, 3])
            e_img=tf.reduce_sum(self.v2*tf.nn.tanh(enc_img_features + decoder_img_features + coverage_features_img), [2, 3])
            
            #Shape of both e and e_img is (batch_size,attn_length)
            attn_dist=self.masked_attention_text(e,enc_mask)
            attn_dist_img=self.masked_attention_img(e_img,img_mask)

            # Update coverage vector
            coverage+=tf.reshape(attn_dist,[batch_size,-1,1,1]) #Shape (batch_size, attn_length, 1 ,1)
            coverage_img+=tf.reshape(attn_dist_img,[batch_size,-1,1,1]) #Shape (batch_size, attn_length, 1 ,1)
        else:
            # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
            e = tf.reduce_sum(self.v1*tf.nn.tanh(enc_features + decoder_features ), [2, 3])
            e_img=tf.reduce_sum(self.v2*tf.nn.tanh(enc_img_features + decoder_img_features ), [2, 3])
            #Shape of both e and e_img is (batch_size,attn_length)
            

            attn_dist=self.masked_attention_text(e,enc_mask)
            attn_dist_img=self.masked_attention_img(e_img,img_mask)
            #Shape of both attn_dist is (batch_size,attn_length)
            # Update coverage vector
            if self.coverage:
                coverage=tf.expand_dims(tf.expand_dims(attn_dist,2),2)
                coverage_img=tf.expand_dims(tf.expand_dims(attn_dist_img,2),2)
        
        context_vector_txt=tf.reduce_sum(tf.reshape(attn_dist,[batch_size, -1, 1, 1])*enc_outputs,[1, 2]) # shape (batch_size, attn_units)
        context_vector_img=tf.reduce_sum(tf.reshape(attn_dist_img,[batch_size, -1, 1, 1])*enc_img_outputs,[1, 2]) # shape (batch_size, attn_units)
        #Calculating Multimodal Context Vector
        decoder_features_new= self.linear3(dec_hidden)
        decoder_img_features_new= self.linear4(dec_hidden)
        #shape of both is now (batch_size,attn_units)
        context_vector_feature=self.linearA(context_vector_txt)
        context_vector_img_feature=self.linearB(context_vector_img)
        beta1=self.v3 * tf.nn.tanh(decoder_features_new + context_vector_feature)
        beta2=self.v4 * tf.nn.tanh(decoder_img_features_new + context_vector_img_feature)
        #Multimodal Context Vector (mm)
        context_vector_mm=beta1 * (context_vector_txt) + beta2 * (context_vector_img)
        context_vector_mm=tf.reshape(context_vector_mm,[-1, self.attn_units])
        #context_vector shape (batch_size, attn_units)
        return context_vector_mm, attn_dist, attn_dist_img, coverage, coverage_img


class Decoder(Layer):
    def __init__(self, embedding_dim, dec_units,embedding_layer, rand_unif_init_mag):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.rand_unif_init = tf.keras.initializers.RandomUniform(-rand_unif_init_mag, rand_unif_init_mag, seed=123)
        ##    embeddings_initializer=tf.keras.initializers.Constant(embeddings_matrix),  # Pre-trained word vector coefficients
        self.embedding = embedding_layer
        self.lstm = LSTM(self.dec_units,return_state=True, recurrent_initializer = 'glorot_uniform', name="decoder-lstm") #For LSTM
        # self.lstm = GRU(self.dec_units,return_state=True,name="decoder-gru",
        #                                recurrent_initializer='glorot_uniform')
        self.linear=Dense(embedding_dim, use_bias=True, name="decoder-linear")

    def call(self, dec_inp, prev_dec_hidden, context_vector):
        #print("Tracing Inside Decoder ..")
        #tf.print("Inside Decoder ..")
        # dec_inp is the decoder input at time-stamp t - shape (batch_size ,1)
        # prev_dec_hidden = (prev_dec_hidden_state, prev_dec_cell_state)

        # dec_inp shape after passing through embedding == (batch_size, 1, embedding_dim)
        dec_inp = self.embedding(dec_inp)
        # shape after concatenation == (batch_size, 1, embedding_dim + 2*enc_units)
        # shape after passing through linear layer ==  (batch_size, 1, embedding_dim)
        dec_inp= self.linear(tf.concat([tf.expand_dims(context_vector, 1), dec_inp], axis=-1))
        # passing the concatenated vector to the GRU
        # output, last_hidden = self.lstm(dec_inp,initial_state=prev_dec_hidden)
        output, last_hidden, last_cell = self.lstm(dec_inp,initial_state=prev_dec_hidden) #For LSTM
        
        #To be used to initialize next decoding stage 
        # dec_states=last_hidden
        dec_states=[last_hidden, last_cell] #For LSTM
        
        
        #dec_states shape== ( batch_size, dec_units)
        # output shape == (batch_size , dec_units)
        
        return dec_inp, output, dec_states


class Prediction(Layer):
    def __init__(self, vocab_size, trunc_norm_init_std):
        super(Prediction,self).__init__()
        self.trunc_norm_init=tf.keras.initializers.TruncatedNormal(stddev=trunc_norm_init_std)
        self.fc = Dense(vocab_size, activation=tf.keras.activations.softmax, name="pred-fc-vocab-dist",
                        kernel_initializer=self.trunc_norm_init, bias_initializer=self.trunc_norm_init)
    
    def call(self,outputs,max_dec_steps):
        #print("Tracing Inside Prediction Layer..")
        #tf.print("Inside Prediction Layer..")
        '''
        outputs : Attention Outputs projected , shape == (batch_size, dec_units) , Tensorarray length = max_dec_steps
        max_dec_steps : max_dec_steps for current input, equals 1 in test mode
        '''
        vocab_dists = tf.TensorArray(tf.float32, size=max_dec_steps)
        for t_step in tf.range(0,limit=max_dec_steps):
            vocab_dists= vocab_dists.write(t_step,self.fc(outputs.read(t_step)))
        #vocab_dist : Tensorarray, length : max_dec_steps, element shape== (batch_size, vocab_size)
        return vocab_dists

class Pointer(Layer):
    def __init__(self):
        super(Pointer, self).__init__()
        self.linear = Dense(1,name="Pointer-linear", use_bias = True)

    def call(self, context_vector, hidden_state,cell_state, dec_inp): #For LSTM
    # def call(self, context_vector, hidden_state, dec_inp):
        #print("Tracing Inside Pointer Layer..")
        #tf.print("Inside Pointer Layer..")
        #Output shape == (batch_size,1)
        return tf.sigmoid(self.linear(tf.concat([context_vector, cell_state, hidden_state, dec_inp],axis = -1)))
        

class Domain_Similarity_Classifier(Layer):
    def __init__(self):
        super(Domain_Similarity_Classifier,self).__init__()
        self.linearh = Dense(1, name="dsc-linear_hidden")
        self.linearc = Dense(1, name="dsc-linear_cell")
        self.linear1 = Dense(1, name = "dsc-linear1")
        self.linear2 = Dense(1, name = "dsc-linear2")

    def call(self,encoder_last_states, img_encoded, sim_img_encoded, dissim_img_encoded):
        #encoder_last_states shape == list of 2 hidden and cell, each of shape (batch_size, enc_units)
        #img_encoder shape == (batch_size, max_img_num , 2*enc_units)
        #sim_img_encoder & dissim_img_encoded shape == (batch_size, max_img_num , embd_dim)
        # img_num_vec shape == batch_size
        # print("Tracing Inside Domain Similarity Classifier..")
        # tf.print("Inside Domain Similarity Classifier..")
        encoder_last_hidden = tf.expand_dims(encoder_last_states[0],1) #shape == (batch_size, 1 , enc_units )
        encoder_last_cell = tf.expand_dims(encoder_last_states[1],1) #shape == (batch_size, 1 , enc_units )
        
        encoder_last_hidden = self.linearh(encoder_last_hidden) #shape == (batch_size, 1 , 1)
        encoder_last_cell = self.linearc(encoder_last_cell)  #shape == (batch_size, 1 , 1)
        img_encoded = self.linear1(img_encoded)  #shape == (batch_size, batch_max_img_num , 1)
        batch_max_img_num = tf.shape(img_encoded)[1]
        sim_img_encoded = self.linear2(sim_img_encoded)[:,:batch_max_img_num,:] # (batch_size, batch_max_img_num , 1)
        dissim_img_encoded = self.linear2(dissim_img_encoded)[:,:batch_max_img_num,:] # (batch_size, batch_max_img_num , 1)
        #Before squeeze shape will be (batch_size, batch_max_img_num, 1)
        
        sim_vecs = tf.squeeze( encoder_last_hidden + encoder_last_cell + img_encoded + sim_img_encoded, axis = -1) #shape == (batch_size, batch_max_img_num)
        dissim_vecs = tf.squeeze(encoder_last_hidden + encoder_last_cell + img_encoded + dissim_img_encoded, axis= -1)
        combined_vecs = tf.concat([sim_vecs, dissim_vecs], axis = -1)
        #shape (batch_size, 2 * batch_max_img_num)
        return combined_vecs



# ### DSC_MSMO Model



class DSC_MSMO(tf.keras.Model):
    def __init__(self, params):
        super(DSC_MSMO, self).__init__()
        self.params = params
        self.attn_units=2*params.enc_units
        self.trunc_norm_init=tf.keras.initializers.TruncatedNormal(stddev=params.trunc_norm_init_std)
        # embeddings_initializer=tf.keras.initializers.Constant(embeddings_matrix),
        self.embedding_layer=Embedding(params.vocab_size, 
                            params.w2v_embed_dim,  
                            embeddings_initializer=self.trunc_norm_init,   
                            trainable=True,
                            name="embedding-layer"   
                            )
        self.encoder = Encoder(params.enc_units,params.dec_units, self.embedding_layer, params.batch_size, params.rand_unif_init_mag, params.trunc_norm_init_std)
        self.image_projector=ImageProjection(params.img_embed_dim,self.attn_units,params.trunc_norm_init_std)
        self.domain_similarity_classifier = Domain_Similarity_Classifier()
        self.attention = MultimodalAttention(self.attn_units, params.coverage, params.mode, params.trunc_norm_init_std )
        self.decoder = Decoder(params.w2v_embed_dim, params.dec_units, self.embedding_layer , params.rand_unif_init_mag)
        self.pointer = Pointer()
        self.predict_dist=Prediction(params.vocab_size,params.trunc_norm_init_std)
        self.conv_w_h=self.add_weight(name="conv-w-h",shape=(1,1,self.attn_units, self.attn_units), trainable=True)
        self.conv_w_h_img = self.add_weight(name="conv-w-h-img",shape=(1,1,self.attn_units, self.attn_units), trainable=True)
        self.linear_attn_output_proj=Dense(params.dec_units, name="linear-attn_output-proj", use_bias=True)
      
    def call_encoder(self, enc_inp):
        # enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp)
        return enc_hidden, enc_output
    
    def call_image_projector(self,enc_img_inp):
        #Projects images embeddings to text encoder output dimensions
        enc_img_projected=self.image_projector(enc_img_inp)
        return enc_img_projected
    
    def call(self, enc_outputs, dec_hidden, enc_mask, enc_extended_inp, enc_img_outputs,img_mask, dec_inp, batch_oov_len, prev_coverage, prev_coverage_img):
        #dec_hidden = [hidden state, cell_state], hence for hidden state dec_hidden[0] is used
        max_dec_steps=tf.shape(dec_inp)[1]
        enc_attn_length=tf.shape(enc_outputs)[1]
        enc_img_attn_length=tf.shape(enc_img_outputs)[1]
        outputs = tf.TensorArray(tf.float32, size=max_dec_steps) #stores the decoder attention projected outputs 
        attentions = tf.TensorArray(tf.float32, size=max_dec_steps, clear_after_read = False)
        attentions_img=tf.TensorArray(tf.float32, size=max_dec_steps, clear_after_read = False)
        p_gens = tf.TensorArray(tf.float32, size=max_dec_steps, clear_after_read = False)
        
        #Reshaping encoder outputs for further downstream tasks
        enc_outputs=tf.expand_dims(enc_outputs,2)
        enc_img_outputs=tf.expand_dims(enc_img_outputs,2)
        #output shape : (batch_size, attn_length,1, attn_units )
        #Extracting features
        enc_features=tf.nn.conv2d(enc_outputs, self.conv_w_h, [1, 1, 1, 1], "SAME")
        enc_img_features=tf.nn.conv2d(enc_img_outputs, self.conv_w_h_img, [1, 1, 1, 1], "SAME")
        #output shape : (batch_size, attn_length,1, attn_units )
        
        #initializing Multimodal context vector
        context_vector=tf.zeros([self.params.batch_size,self.attn_units]) # shape == (batch_size , attn_units)
        coverage_set=tf.constant(False) #Keeps track of whether coverage vector is none or not, False means None , True means holds some value
        if self.params.mode == "test" and self.params.coverage: # True only in decode mode
            coverage = tf.expand_dims(tf.expand_dims(prev_coverage,2),3) #shape == (batch_size,attn_length,1,1)
            coverage_img = tf.expand_dims(tf.expand_dims(prev_coverage_img,2),3) #shape == (batch_size,attn_length,1,1)
            coverage_set=tf.constant(True)
        else:
            coverage=tf.zeros([self.params.batch_size,enc_attn_length,1,1]) # shape == (batch_size,attn_length,1,1)
            coverage_img=tf.zeros([self.params.batch_size,enc_img_attn_length,1,1]) # shape == (batch_size,attn_length,1,1)
        
        if self.params.initial_state_attention: # True only in decode mode
            context_vector, _ , _ , coverage, coverage_img=self.attention(coverage_set, dec_hidden, enc_outputs, enc_features, enc_mask, coverage,enc_img_outputs, enc_img_features, img_mask, coverage_img)
        for t_step in tf.range(0,limit=max_dec_steps):
            #t_step stands for timestep
            #tf.print("Timestep :",t_step)
            dec_x, dec_out, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t_step],1), dec_hidden, context_vector)
            #Here dec_x refers to the concatenated decoder inp vector with the context vector
            #dec_hidden=[hidden state, cell_state]
            if tf.math.equal(t_step,tf.constant(0,dtype=tf.int32)) and self.params.initial_state_attention:
                context_vector, attn_dist, attn_dist_img, _, _=self.attention(coverage_set, dec_hidden, enc_outputs, enc_features, enc_mask, coverage,enc_img_outputs, enc_img_features, img_mask, coverage_img)
            else:
                if self.params.coverage and tf.greater(t_step,tf.constant(0)): # For Non decode mode , coverage gets set from t_step =1 onwards
                    coverage_set = tf.constant(True)
                context_vector, attn_dist, attn_dist_img, coverage, coverage_img=self.attention(coverage_set, dec_hidden, enc_outputs, enc_features, enc_mask, coverage,enc_img_outputs, enc_img_features, img_mask, coverage_img)
            
            attentions= attentions.write(t_step,attn_dist)
            attentions_img= attentions_img.write(t_step,attn_dist_img)
            p_gen = self.pointer(context_vector, dec_hidden[0], dec_hidden[1], tf.squeeze(dec_x, axis=1)) 
            p_gens= p_gens.write(t_step,p_gen)
            #dec_out shape == (batch_size, dec_units)
            #context_vector shape == (batch_size, attn_units)
            output=self.linear_attn_output_proj(tf.concat([dec_out,context_vector],axis=-1))
            #outputs shape== (batch_size, dec_units)
            outputs= outputs.write(t_step,output)
        
        # outputs shape == (batch_size,dec_units) len = dec_steps , tensorarray
        # attentions shape == (batch_size,attn_length) len = dec_steps , tensorarray
        # attentions_img shape == (batch_size,attn_length) len = dec_steps , tensorarray
        # p_gens shape == (batch_size,1) len = dec_steps , tensorarray
        
        vocab_dists=self.predict_dist(outputs, max_dec_steps)
        # vocab_dists shape == (batch_size,vocab_size) len = dec_steps , tensorarray

        final_dists = calc_final_dist( enc_extended_inp, vocab_dists, attentions, p_gens, batch_oov_len, self.params.vocab_size, self.params.batch_size)
        #final_dists is a tensorarray of length = dec_steps and each content vector is of shape (batch_size, extended_vocab_size)

        if self.params.coverage and coverage_set:
            coverage = tf.reshape(coverage, [self.params.batch_size, -1]) #shape (batch_size , attn_length)
            coverage_img = tf.reshape(coverage_img, [self.params.batch_size, -1]) #shape (batch_size , attn_length)
        
        if self.params.mode == "train" or self.params.mode == "eval":
            return final_dists, attentions, attentions_img
        else:
            return tf.transpose(final_dists.stack(), perm=[1,0,2]), tf.transpose(attentions.stack(), perm=[1,0,2]) , tf.transpose(attentions_img.stack(), perm=[1,0,2]), tf.transpose(p_gens.stack(), perm=[1,0,2]), dec_hidden, context_vector, coverage, coverage_img
