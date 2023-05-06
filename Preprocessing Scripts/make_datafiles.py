import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
# from tensorflow.core.example import example_pb2

#Set path of stanford nlp
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


os.environ["CLASSPATH"]="/DATA/sourajit_2011mc14/project/software/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar" 
dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

DATA_FOLDER="../../data/"
all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

msmo_dir=os.path.join(DATA_FOLDER,"msmo2")
tokenized_articles_dir = os.path.join(msmo_dir,"tokenized")
finished_files_dir = os.path.join(msmo_dir,"finished_files")
chunks_dir = os.path.join(finished_files_dir, "chunked")


VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


def chunk_file(set_name):
  in_file = os.path.join(finished_files_dir,'%s.bin' % set_name)
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  # for set_name in ['train', 'val', 'test']:
  for set_name in [ 'train', 'val', 'test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)


def tokenize_articles(articles_dir, tokenized_articles_dir):
  """Maps a whole directory of .article files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." % (articles_dir, tokenized_articles_dir))
  articles = os.listdir(articles_dir)
  # make IO list file
  print("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in articles:
      f.write("%s \t %s\n" % (os.path.join(articles_dir, s), os.path.join(tokenized_articles_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print("Tokenizing %i files in %s and saving in %s..." % (len(articles), articles_dir, tokenized_articles_dir))
  subprocess.call(command)
  print("Stanford CoreNLP Tokenizer has finished.")
  os.remove("mapping.txt")

  # # Check that the tokenized articles directory contains the same number of files as the original directory
  # num_orig = len(os.listdir(articles_dir))
  # num_tokenized = len(os.listdir(tokenized_articles_dir))
  # if num_orig != num_tokenized:
    # raise Exception("The tokenized articles directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_articles_dir, num_tokenized, articles_dir, num_orig))
  print("Successfully finished tokenizing %s to %s.\n" % (articles_dir, tokenized_articles_dir))


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def get_hash(url):
  url_b=url.encode('utf-8')
  return hashhex(url_b.strip())

#Hash function to hash the url to generate file name
def hashhex(s): 
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_url_hashes(url_list):
  return [get_hash(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@title" in line: return line
  if "@body" in line: return line
  if "@summary" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(article_file):
  lines = read_text_file(article_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  summary = []
  next_is_summary = False
  next_is_article=False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@body"):
      next_is_summary = False
      next_is_article=True
    elif line.startswith("@summary"):
      next_is_article=False
      next_is_summary = True
    elif next_is_summary:
      summary.append(line)
    elif next_is_article:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in summary])

  return article, abstract


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_to_bin(url_file, out_file, makevocab=False):
  """Reads the tokenized .article files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print("Making bin file for URLs listed in %s..." % url_file)
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  article_fnames = [a+".txt" for a in url_hashes]
  num_articles = len(article_fnames)

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx,a in enumerate(article_fnames):
      if idx % 1000 == 0:
        print("Writing article %i of %i; %.2f percent done" % (idx, num_articles, float(idx)*100.0/float(num_articles)))

      # Look in the tokenized article dirs to find the .article file corresponding to this url
      
      article_file = os.path.join(tokenized_articles_dir, a)
      
      # Get the strings to write to .bin file
      article, abstract = get_art_abs(article_file)

      # Write to tf.Example
      feature = {
        'article': _bytes_feature(article.encode('utf-8')),
        'abstract': _bytes_feature(abstract.encode('utf-8'))
      }
      example_proto=tf.train.Example(features=tf.train.Features(feature=feature))
      tf_example_str = example_proto.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")


def check_num_articles(articles_dir, num_expected):
  num_articles = len(os.listdir(articles_dir))
  if num_articles != num_expected:
    raise Exception("articles directory %s contains %i files but should contain %i" % (articles_dir, num_articles, num_expected))


if __name__ == '__main__':
  

  # Check the articles directories contain the correct number of .article files
  check_num_articles(cnn_articles_dir, num_expected_cnn_articles)
  check_num_articles(dm_articles_dir, num_expected_dm_articles)

  # Create some new directories
  if not os.path.exists(msmo_dir): os.makedirs(msmo_dir)
  if not os.path.exists(tokenized_articles_dir): os.makedirs(tokenized_articles_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  ##For training set there are 21 directories
  for i in range(1,21):
    articles_dir=os.path.join(DATA_FOLDER,"data{}".format(i),"article")
    # Run stanford tokenizer on both articles dirs, outputting to tokenized articles directories
    tokenize_articles(articles_dir, tokenized_articles_dir)

#   #For validation sets
  valid_articles_dir=os.path.join(DATA_FOLDER,"valid_data","article")
  tokenize_articles(valid_articles_dir, tokenized_articles_dir)
  #For test data sets
  test_articles_dir=os.path.join(DATA_FOLDER,"test_data","article")
  tokenize_articles(test_articles_dir, tokenized_articles_dir)

  # Read the tokenized articles, do a little postprocessing then write to bin files
  write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"))
  write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
  write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"), makevocab=True)

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all()