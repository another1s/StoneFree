import numpy as np
from gensim.models import fasttext
from gensim.test.utils import datapath
import tensorflow as tf
cap_path = datapath("crime-and-punishment.bin")
fb_model = fasttext.load_facebook_model(cap_path)
word_index = []
vectors = fb_model.wv.vectors_vocab
vocab = fb_model.wv.vocab
word_index.append(vocab)

data = [[1,2,3],[4,5,6]]
data_np = np.asarray(data, np.float32)

data_tf = tf.convert_to_tensor(vectors, np.float32)

input_embedded = tf.nn.embedding_lookup(data_tf, [1, 3])
sess = tf.InteractiveSession()
print(data_tf.eval())
print(input_embedded.eval())
sess.close()