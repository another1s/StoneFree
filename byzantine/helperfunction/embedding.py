from gensim.models import fasttext

default_embedding = ['word2vec', 'FastText']


class NotoriousBig:
    def __init__(self, embedding_switch: 'str', embedding_addr: 'str', training_data_address: 'str'):
        self.word_embedding = self.read_embedding(embedding_switch, embedding_addr)
        self.data, self.labels, self.pos_tag = self.read_data(training_data_address)

    @staticmethod
    def update_model(model, name):
        model.save('../models/embedding/crawl-300d-2M-subword_' + name + '.bin')

    @staticmethod
    def read_embedding(switch, data_addr):
        if switch == 'FastText':
            fb_model = fasttext.load_facebook_model(path=data_addr)
            return fb_model

    @staticmethod
    def read_data(data_address):
        words_address = data_address + 'words.txt'
        labels_address = data_address + 'label.txt'
        tag_address = data_address + 'pos_tag.txt'
        with open(words_address, 'r', encoding='GBK') as f:
            words = f.readlines()
            f.close()
        with open(labels_address, 'r', encoding='utf-8') as f:
            labels = f.readlines()
            f.close()
        with open(tag_address, 'r', encoding='utf-8') as f:
            pos_tag = f.readlines()
            f.close()
        return words, labels, pos_tag

    def add_new_word(self, new_words, model):
        model.build_vocab(new_words, update=True)
        model.train(sentences=new_words, total_examples=len(new_words), epochs=30)
        self.word_embedding = model
        self.update_model(model, str(len(new_words)))

# config = {'switch': 'FastText', 'addr': '../models/embedding/crawl-300d-2M-subword.bin'}
# n = NotoriousBig(embedding_switch=config['switch'], embedding_addr=config['addr'])
# if 'equity' in n.word_embedding.wv.vocab:
#     print(n.word_embedding.wv['equity'])