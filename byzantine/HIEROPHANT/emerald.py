from byzantine.HIEROPHANT.Ner import BilstmNer
from byzantine.helperfunction.embedding import NotoriousBig

import warnings
import tensorflow as tf


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = dict()
    config['switch'] = 'FastText'
    config['addr'] = '../models/embedding/crime-and-punishment.bin'
    config['train_addr'] = '../data/input_data/'
    metadata = NotoriousBig(embedding_switch=config['switch'], embedding_addr=config['addr'],
                            training_data_address=config['train_addr'])
    # data_train = BatchGenerator(X=metadata.data, y=metadata.labels, shuffle=True)
    for sentence in metadata.data:
        metadata.check_words(words=sentence)
    metadata.add_new_word(new_words=metadata.newly_imported, model=metadata.word_embedding)

    splash = {
        'learning_rate': 0.001,
        'batch_size': 10,
        'embedding_dimension': 300,
        'embedding_size': 2000000,
        'sentence_length': 128,
        'pretrained_model': '',
        'tag_size': 4
    }

    Bilstm_Crf = BilstmNer(config=splash, embedding_pretrained=metadata.word_embedding)
    data = Bilstm_Crf.fasttext_embedding_conversion(mode='fasttext', data=metadata.data)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()
        Bilstm_Crf.train(sess=sess, training_data=data, training_labels=metadata.labels, epoches=20, batch_size=10)
    sess.close()
    print('done')





