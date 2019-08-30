import numpy as np
from gensim.models import fasttext
from gensim.test.utils import datapath
import tensorflow as tf



w = ['erere', 'weerr', 'aaaaa']

Tokenlizer = tf.keras.preprocessing.text.Tokenizer()
Tokenlizer.fit_on_texts(w)
print(Tokenlizer.get_config())




def halo(a, b):
    return a+1, b+1
# a = np.array([2, 4, 5, 6])
# for i in a:
#     print(i)
# c = np.ones(4) * 9
# b = np.pad(a, (2, 3), mode='mean')
# print(a.shape)
a = np.array([32,32,23232.3,77])
print(a[0:3,])
b= [3,1]
print(list(map(lambda x:x+1, a)))
#print(list(map(lambda x, y:halo(a=3, b=7), [(i, j) for i, j in zip(a, b)])))

# cap_path = datapath("crime-and-punishment.bin")
# fb_model = fasttext.load_facebook_model(cap_path)
# word_index = []
# vectors = fb_model.wv.vectors_vocab
# vocab = fb_model.wv.vocab
# word_index.append(vocab)
#
# data = [[1,2,3],[4,5,6]]
# data_np = np.asarray(data, np.float32)
#
# data_tf = tf.convert_to_tensor(vectors, np.float32)
#
# input_embedded = tf.nn.embedding_lookup(data_tf, [1, 3])
# sess = tf.InteractiveSession()
# print(data_tf.eval())
# print(input_embedded.eval())
# sess.close()