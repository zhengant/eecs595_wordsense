from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#########################################################################
# most of this code is adapted from extract_features.py from the Google 
# BERT github repository
# https://github.com/google-research/bert
#########################################################################

import os
import codecs
import collections
import json
import re

# assumes you have cloned the BERT repository to your working directory
import bert.modeling
import bert.tokenization
import numpy as np
import tensorflow as tf
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

def init_tf_flags():
  flags = tf.flags
  bert_model_dir = 'uncased_L-12_H-768_A-12'

  flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

  flags.DEFINE_string(
      "bert_config_file", bert_model_dir + '/bert_config.json',
      "The config json file corresponding to the pre-trained BERT model. "
      "This specifies the model architecture.")

  flags.DEFINE_integer(
      "max_seq_length", 128,
      "The maximum total input sequence length after WordPiece tokenization. "
      "Sequences longer than this will be truncated, and sequences shorter "
      "than this will be padded.")

  flags.DEFINE_string(
      "init_checkpoint", bert_model_dir + '/bert_model.ckpt',
      "Initial checkpoint (usually from a pre-trained BERT model).")

  flags.DEFINE_string("vocab_file", bert_model_dir + '/vocab.txt',
                      "The vocabulary file that the BERT model was trained on.")

  flags.DEFINE_bool(
      "do_lower_case", True,
      "Whether to lower case the input text. Should be True for uncased "
      "models and False for cased models.")

  flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

  flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

  flags.DEFINE_string("master", None,
                      "If using a TPU, the address of the master.")

  flags.DEFINE_integer(
      "num_tpu_cores", 8,
      "Only used if `use_tpu` is True. Total number of TPU cores to use.")

  flags.DEFINE_bool(
      "use_one_hot_embeddings", False,
      "If True, tf.one_hot will be used for embedding lookups, otherwise "
      "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
      "since it is much faster.")


class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = bert.modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = bert.modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    all_layers = model.get_all_encoder_layers()

    predictions = {
        "unique_id": unique_ids,
    }

    for (i, layer_index) in enumerate(layer_indexes):
      predictions["layer_output_%d" % i] = all_layers[layer_index]

    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (example.unique_id))
      tf.logging.info("tokens: %s" % " ".join(
          [bert.tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_input_file(tsv_filename):
  sentences = []
  metadata = []
  unique_id = 0
  with open(tsv_filename, 'r') as input:
    while True:
      line = bert.tokenization.convert_to_unicode(input.readline())
      if not line:
        break
      
      text_id, lemma, pos, instance, sentence = line.split('\t')
      sentence = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", sentence)
      if m is None:
        text_a = sentence
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      sentences.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1

      metadata.append((text_id, lemma, pos, instance))

  return sentences, metadata


def embed_sentences_in_file(tsv):
  sentences, metadata = read_input_file(tsv)
  embeddings = []

  # setup
  FLAGS = tf.flags.FLAGS

  layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

  bert_config = bert.modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tokenizer = bert.tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      layer_indexes=layer_indexes,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.batch_size)

  features = convert_examples_to_features(
    examples=sentences, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)
  unique_id_to_feature = {}
  for feature in features:
    unique_id_to_feature[feature.unique_id] = feature

  input_fn = input_fn_builder(
    features=features, seq_length=FLAGS.max_seq_length) 

  # get embedding for each instance of the word
  for idx, result in enumerate(estimator.predict(input_fn, yield_single_examples=True)):
    unique_id = int(result["unique_id"])
    feature = unique_id_to_feature[unique_id]

    for i, token in enumerate(feature.tokens):
      if token == metadata[idx][3]:
        layers = []
        for j, _ in enumerate(layer_indexes):
          layer_output = result['layer_output_%d' % j]
          layers.append([
            round(float(x), 6) for x in layer_output[i:(i+1)].flat])

        # concatenate last x layers together
        embeddings.append(np.concatenate(layers, axis=None))
        break

  return embeddings, metadata


def find_closest_inlier(i, labels, distances):
  closest = np.argsort(distances[i])
  for idx in closest:
    if labels[idx] >= 0:
      return idx

  return 0


def cluster_embeddings_dbscan(distances, eps, min_samples):
  db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
  labels = db.fit_predict(distances)

  # assign noisy points the label of their nearest non-noisy point
  for i in range(len(labels)):
    if labels[i] == -1:
      labels[i] = labels[find_closest_inlier(i, labels, distances)]

  return labels


def compute_embedding_distances(embeddings):
  normalized_embeddings = np.array(embeddings)
  for i in range(len(normalized_embeddings)):
    normalized_embeddings[i] = np.divide(normalized_embeddings[i], np.linalg.norm(normalized_embeddings[i]))

  return pairwise_distances(normalized_embeddings, metric='euclidean', n_jobs=-1)


def output_senses(labels, metadata, outfile):
  label_to_sense = {}
  with open(outfile, 'a') as out:
    for i, label in enumerate(labels):
      out.write(metadata[i][1] + '.' + metadata[i][2] + ' ')
      out.write(metadata[i][0] + ' ')
      if label not in label_to_sense:
        label_to_sense[label] = len(label_to_sense) + 1
      
      out.write(str(label_to_sense[label]) + '/1.0\n')
      

def cluster_all_words(tsv_filenames, tsv_dir, eps, min_samples, outfile):
  # setup
  tf.logging.set_verbosity(tf.logging.WARN)
  init_tf_flags()

  # read files
  for tsv in tsv_filenames:
    embeddings, metadata = embed_sentences_in_file(tsv_dir + '/' + tsv)
    distances = compute_embedding_distances(embeddings)
    print(distances)
    print(np.mean(distances))
    print(np.max(distances))

    labels = cluster_embeddings_dbscan(distances, eps, min_samples)
    print(labels)

    output_senses(labels, metadata, outfile)


def main():
  tsv_dir = 'Datasets'
  tsv_filenames = os.listdir(tsv_dir)
  # tsv_filenames = ['Datasets/add.tsv']

  cluster_all_words(tsv_filenames, tsv_dir, 0.5, 2, 'senses.out')


if __name__ == '__main__':
  main()
    