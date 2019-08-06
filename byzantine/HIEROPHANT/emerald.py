from byzantine.HIEROPHANT.Ner import BilstmNer
from byzantine.helperfunction.embedding import NotoriousBig
from byzantine.helperfunction.data_split import BatchGenerator

if __name__ == '__main__':
    config = dict()
    config['switch'] = 'JustText'
    config['addr'] = '../models/embedding/crawl-300d-2M-subword.bin'
    config['train_addr'] = '../data/input_data/'
    metadata = NotoriousBig(embedding_switch=config['switch'], embedding_addr=config['addr'],
                            training_data_address=config['train_addr'])
    data_train = BatchGenerator(X=metadata.data, y=metadata.labels, shuffle=True)

    splash = {
        'learning_rate': 0.001,
        'batch_size': 10,
        'embedding_dimension': 300,
        'embedding_size': 2000000,
        'sentence_length': 128,
        'tag_size': 4
    }





