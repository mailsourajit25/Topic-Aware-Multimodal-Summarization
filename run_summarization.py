import os
import sys
# os.chdir(os.path.join(os.getcwd(),"project","code"))
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import argparse
import logging
from data import Vocab
import shutil
from utils import get_latest_model, calc_running_avg_loss
from train_test_eval import train_model, test_and_save, eval_model
from batcher import batcher as train_batcher
from test_eval_batcher import batcher as test_eval_batcher
from model import DSC_MSMO


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def del_logs():
    '''Deletes all previous checkpoints and logs. Call this only when needed'''
    if(os.path.exists(params.checkpoint_dir)):
        shutil.rmtree(params.checkpoint_dir)
    if(os.path.exists(params.tfsummary_logdir)):
        shutil.rmtree(params.tfsummary_logdir)
    if(os.path.exists(params.log_root)):
        shutil.rmtree(params.log_root)
    
def restore_best_model(params):
    '''Restores best model from eval directory to train directory. Use if while training you are getting NaN or your model is overfitting'''
    model = DSC_MSMO(params)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0),DSC_MSMO=model)
    eval_ckpt_dir= os.path.join(params.log_root, 'eval_ckpt')
    eval_ckpt_manager = tf.train.CheckpointManager(ckpt, eval_ckpt_dir, max_to_keep=3)
    train_ckpt_manager = tf.train.CheckpointManager(ckpt, params.checkpoint_dir, max_to_keep=3)
    ckpt.restore(eval_ckpt_manager.latest_checkpoint).expect_partial()
    train_ckpt_manager.save(checkpoint_number=int(ckpt.step))
    print(f"Restored best model checkpoint - {eval_ckpt_manager.latest_checkpoint}")

    



### Eval Function
def eval(params, logger):
    assert params.mode.lower() == "eval", "change training mode to 'test'"
    
    print("Creating the vocab ...")
    vocab = Vocab(params.vocab_path, params.vocab_size, logger)

    # embeddings_matrix = get_embedding(params.vocab_size, params.w2v_embed_dim, vocab, params.w2v_embedding_path, params)

    logger.info("Loading the model ...")
    model = DSC_MSMO(params)

    print("Creating the batcher ...")
    b = test_eval_batcher(params.data_path, params.img_feature_path, vocab)
    batched_dataset=b.get_batched_dataset(params, params.batch_size, logger)
    batch_iterator = iter(batched_dataset)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}".format(params.checkpoint_dir)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0),DSC_MSMO=model)
    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    
    # Save bestmodel checkpoint in eval dir
    eval_ckpt_dir = os.path.join(params.log_root, 'eval_ckpt')
    if not os.path.exists(eval_ckpt_dir):
        os.makedirs(eval_ckpt_dir)

    eval_ckpt_manager = tf.train.CheckpointManager(ckpt, eval_ckpt_dir, max_to_keep=3)

    running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None  # will hold the best loss achieved so far
    prev_coverage = tf.zeros([params.batch_size, params.max_enc_steps])
    prev_coverage_img = tf.zeros([params.batch_size, params.max_img_num])
    
    #summary writer
    # eval_summary_writer = tf.summary.create_file_writer(params.tfsummary_logdir)
    template='Mode : eval, Exp: {} ,Step {}, Running_avg_loss {:.4f}, Text_Loss {:.4f}'
    while True:
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
        latest_ckpt = get_latest_model(ckpt_manager)
        if latest_ckpt!="":
            path = latest_ckpt
        else:
            path = ckpt_manager.latest_checkpoint
        
        ckpt.restore(path).expect_partial()
        print(f"\nModel restored - {path}")

        batch = batch_iterator.get_next()
        
        loss = eval_model(model, batch, params, prev_coverage, prev_coverage_img)

        # calculate running avg loss
        running_avg_loss = calc_running_avg_loss(loss[0].numpy(), running_avg_loss)
        print(template.format(params.experiment,int(ckpt.step),running_avg_loss,loss[0].numpy()))
        # with eval_summary_writer.as_default():
        #     tf.summary.scalar('text_loss',loss[0].numpy(), step=int(ckpt.step))
        #     tf.summary.scalar('running_avg_loss',running_avg_loss, step=int(ckpt.step))
        #     if params.coverage:
        #         print("Coverage_Loss {:.4f}, Coverage_Img_Loss {:.4f}".format(loss[1].numpy(),loss[2].numpy()))
        #         tf.summary.scalar('coverage_loss',loss[1].numpy(), step=int(ckpt.step))
        #         tf.summary.scalar('coverage_loss_img',loss[2].numpy(), step=int(ckpt.step))
        
        if best_loss is None or running_avg_loss < best_loss:
            logger.info(f'Found new best model with {running_avg_loss:.3f} running_avg_loss. Saving to {eval_ckpt_dir}')
            eval_ckpt_manager.save(checkpoint_number=int(ckpt.step))
            best_loss = running_avg_loss


### Test Function

def test(params,logger):
    assert params.mode.lower() == "test", "change training mode to 'test'"
    assert params.test_save_dir, "Please provide a dir where to save the results"
    assert params.beam_size == params.batch_size, "Beam size must be equal to batch_size, change the params"
    assert params.test_csv_path and params.test_csv_path.endswith(".csv"), "Please enter valid test_csv_path"
    assert params.test_img_path , "Please enter valid test_img_path"
    print("Creating the vocab ...")
    vocab = Vocab(params.vocab_path, params.vocab_size, logger)

    # embeddings_matrix = get_embedding(params.vocab_size, params.w2v_embed_dim, vocab, params.w2v_embedding_path, params)

    logger.info("Loading the model ...")
    model = DSC_MSMO(params)

    print("Creating the batcher ...")
    b = test_eval_batcher(params.data_path, params.img_feature_path, vocab)
    batched_dataset=b.get_batched_dataset(params, params.batch_size, logger)
    batch_iterator = iter(batched_dataset)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}".format(params.checkpoint_dir)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0),DSC_MSMO=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    latest_ckpt = get_latest_model(ckpt_manager)
    if latest_ckpt!="":
        path = latest_ckpt
    else:
        path = ckpt_manager.latest_checkpoint

    ckpt.restore(path).expect_partial()
    print(f"Model restored - {path}")

    test_and_save(params, logger, batch_iterator, model, vocab, latest_ckpt)



    



def train(params,logger):
    '''Function to start training the model'''
    assert params.mode.lower() == "train", "change training mode to 'train'"

    print("Creating the vocab from :", params.vocab_path)
    vocab = Vocab(params.vocab_path, params.vocab_size , logger)

    # print("Creating the embedding_matrix from:",params.w2v_embedding_path)
    # embeddings_matrix = get_embedding(params.vocab_size, params.w2v_embed_dim, vocab, params.w2v_embedding_path, params)
    

    logger.info("Building the model ...")
    model = DSC_MSMO(params)

    print("Creating the batcher ...")
    b = train_batcher(params.data_path, params.img_feature_path, params.sim_img_feature_path, params.dissim_img_feature_path, vocab)
    batched_dataset=b.get_batched_dataset(params, params.batch_size, logger)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}".format(params.checkpoint_dir)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), DSC_MSMO=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    # print(ckpt_manager.checkpoints)
    latest_ckpt = get_latest_model(ckpt_manager)
    if latest_ckpt!="":
        ckpt.restore(latest_ckpt)
        print("Restored from {}".format(latest_ckpt))
    else:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Initializing from scratch.")
    logger.info("Starting the training ...")
    
    train_model(model, batched_dataset, params, ckpt, ckpt_manager)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DSC_MSMO Model Argument Parser")
    parser.add_argument("--data_path",dest="data_path", type=str, required=True, help="Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.")
    parser.add_argument("--img_feature_path", dest="img_feature_path", type=str, required=True, help="Path expression to img feature datafiles.")
    parser.add_argument("--vocab_path", dest="vocab_path", type=str, required=True, help="Path expression to text vocabulary file.")
    parser.add_argument("--sim_img_feature_path", dest="sim_img_feature_path", type=str, help="Path expression to similar img feature datafiles.")
    parser.add_argument("--dissim_img_feature_path", dest="dissim_img_feature_path", type=str, help="Path expression to dissimilar img feature datafiles.")
    

    parser.add_argument('--mode',dest='mode',type=str,default= 'train',help= 'must be one of train/eval/decode')
    parser.add_argument('--log_root',dest='log_root',type=str,default= 'logs',help= 'Root directory for all logging.')
    parser.add_argument('--experiment',dest='experiment',type=str,default= 'msmo',help= 'Name for experiment. Logs will be saved in a directory with this name')
    parser.add_argument('--tfsummary_logdir',dest='tfsummary_logdir',type=str,default= 'summary_logs',help= 'Root directory for all tensorboad logging details.')
    parser.add_argument('--test_save_dir',dest='test_save_dir',type=str,default= 'test_save/',help= 'Directory to save all the results after testing')
    parser.add_argument('--w2v_embedding_path',dest='w2v_embedding_path',type=str,default= '',help= 'If pretrained embedding is passed, then we pass the path')
    parser.add_argument('--test_csv_path',dest='test_csv_path',type=str,default= '',help= 'Path to annotated test set csv')

    parser.add_argument('--enc_units',dest='enc_units',type=int,default= 256,help= 'dimension of encoder RNN hidden states')
    parser.add_argument('--dec_units',dest='dec_units',type=int,default= 256,help= 'dimension of decoder RNN hidden states')
    parser.add_argument('--w2v_embed_dim',dest='w2v_embed_dim',type=int,default= 128,help= 'dimension of word embeddings')
    parser.add_argument('--img_embed_dim',dest='img_embed_dim',type=int,default= 4096,help= 'dimension of image embeddings')
    parser.add_argument('--batch_size',dest='batch_size',type=int,default= 16,help= 'minibatch size')
    parser.add_argument('--max_enc_steps',dest='max_enc_steps',type=int,default= 400,help= 'max timesteps of encoder (max source text tokens)')
    parser.add_argument('--max_dec_steps',dest='max_dec_steps',type=int,default= 100,help= 'max timesteps of decoder (max summary tokens)')
    parser.add_argument('--max_img_num',dest='max_img_num',type=int,default= 10,help= 'Maximum number of images considered per article')
    parser.add_argument('--beam_size',dest='beam_size',type=int,default= 4,help= 'beam size for beam search decoding.')
    parser.add_argument('--min_dec_steps',dest='min_dec_steps',type=int,default= 35,help= 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
    parser.add_argument('--vocab_size',dest='vocab_size',type=int,default= 50000,help= 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number')
    parser.add_argument('--max_train_iterations',dest='max_train_iterations',type=int,default= 100000,help= 'maximum train steps')
    parser.add_argument('--num_to_test',dest='num_to_test',type=int,default= 5,help= 'Number of samples to be tested')
    parser.add_argument('--num_to_eval',dest='num_to_eval',type=int,default= 5,help= 'Number of samples to be evaluated')

    parser.add_argument('--checkpoint_dir',dest='checkpoint_dir',type=str,default= "checkpoint_dir/",help= 'Directory to save model checkpoints')
    parser.add_argument('--checkpoint_save_steps',dest='checkpoint_save_steps',type=int, default= 32,help= 'Number of steps after which checkpoints needs to saved')
    parser.add_argument('--test_save_steps',dest='test_save_steps',type=int, default= 500,help= 'Number of steps after which test results will be saved for attention visualization')
    parser.add_argument('--test_img_path',dest='test_img_path',type=str, default='' ,help= 'Path of test image directory for visualization in web interface')
    parser.add_argument('--restore_best_model', default=False, action='store_true', help='Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')
    

    parser.add_argument('--lr',dest='lr',type=float,default= 0.15,help= 'learning rate')
    parser.add_argument('--adagrad_init_acc',dest='adagrad_init_acc',type=float,default= 0.1,help= 'initial accumulator value for Adagrad')
    parser.add_argument('--rand_unif_init_mag',dest='rand_unif_init_mag',type=float,default= 0.02,help= 'magnitude for lstm cells random uniform inititalization')
    parser.add_argument('--trunc_norm_init_std',dest='trunc_norm_init_std',type=float,default= 1e-4,help= 'std of trunc norm init')
    parser.add_argument('--max_grad_norm',dest='max_grad_norm',type=float,default= 2.0,help= 'for gradient clipping')

    parser.add_argument('--pointer_gen', default=True, action='store_true', help='If True, use pointer-generator model. If False, use baseline model.')
    parser.add_argument('--coverage', default=False, action='store_true', help='Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
    parser.add_argument('--initial_state_attention', default=False, action='store_true', help='It is true only during decode mode')

    parser.add_argument('--single_pass', default=False, action='store_true', help='For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
    parser.add_argument('--cov_loss_wt',dest='cov_loss_wt',type=float,default= 1.0,help= 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
    parser.add_argument('--cov_loss_wt_img',dest='cov_loss_wt_img',type=float,default= 1.0,help= 'Weight of image coverage loss . If zero, then no incentive to minimize coverage loss.')
    parser.add_argument('--classifier_wt',dest='classifier_wt',type=float,default= 1.0,help= 'Weight of domain similarity classifier loss')
    
    
    params = parser.parse_args()
    params.initial_state_attention = tf.constant(params.initial_state_attention)
    params.coverage = tf.constant(params.coverage)
    #Optional changes in configuration paths(can be removed if not needed)
    params.checkpoint_dir = os.path.join(params.log_root,params.checkpoint_dir,params.experiment)
    params.test_save_dir = os.path.join(params.log_root,params.test_save_dir,params.experiment)
    params.tfsummary_logdir = os.path.join(params.log_root,params.tfsummary_logdir,params.experiment,params.mode)

    #Setting up logger and log_root folder
    if not os.path.exists(params.log_root):
        os.makedirs(params.log_root)


    logFormatter = logging.Formatter('%(asctime)s - %(name)s:%(levelname)s - %(message)s')
    logger = logging.getLogger()

    fileHandler = logging.FileHandler(os.path.join(params.log_root,f"{params.mode} - {params.experiment}.log"),mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.DEBUG)

    logger.info(f"Process Id = {os.getpid()}")
    logger.info(params)

    tf.random.set_seed(111) # a seed value for randomness

    if(params.restore_best_model):
        restore_best_model(params)


    if params.mode == "train":
        train(params,logger)
    elif params.mode == "eval":
        eval(params,logger)
    else:
        #Parameters for decode or test mode
        # params.coverage = tf.constant(True)
        params.initial_state_attention = tf.constant(True) #always true in decode mode
        if not os.path.exists(params.test_save_dir):
            os.makedirs(params.test_save_dir)
        #For beam search decoding batch size equals beam size
        params.batch_size = params.beam_size
        test(params,logger)
    


