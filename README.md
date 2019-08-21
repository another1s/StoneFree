# StoneFree
working on nlp..
## Project structure
* data : training data and testing data 
* helpfunction: data preprocessing function
* models: pretrained word-embedding model and neural network 
* HIEROPHANT
    * Dnn.py : stupid structured data classifier
    * emerald.py: [main entrance for Ner](https://github.com/another1s/StoneFree/blob/master/byzantine/HIEROPHANT/emerald.py)
    * Ner.py: literally 
    * teee.py: test script
    * features_eng.py: for future feature engineering use
## how to install
clone project: git clone https://github.com/another1s/StoneFree.git 

download fasttext pretrained: wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip. 
can be replaced by any fasttext alternatives  

## Issues about deploying this project
1. numpy version == 1.16.4 to avoid warning