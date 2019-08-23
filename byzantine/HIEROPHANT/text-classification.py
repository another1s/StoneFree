import tensorflow as tf
import numpy as np


class Sa:
    def __init__(self, config, embedding_pretrained, void_index, dropout_keep=1):
        # the keys of config:
        # {
        #     lr: learning rate (float)
        #     batch_size : number of instances being processed per iteration (int)
        #     embedding_dim: embedding dimension ( vector dimensions ) (int)
        #     embedding_size: number of words in dictionary   (int)
        #     sen len: sentence length     (int)
        #     tag_size: how many different name entity  (int)
        # }
        self.lr = config['learning_rate']
        self.batch_size = config['batch_size']
        self.embedding_dim = config['embedding_dimension']
        self.embedding_size = config['embedding_size']
        self.sentence_len = config['sentence_length']
        self.pretrained_model = config['pretrained_model']
        self.tag_size = config['tag_size']
        self.config = config

    def model(self):

        return

    def train(self):

        return

    def test(self):

        return