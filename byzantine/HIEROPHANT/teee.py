import numpy as np
from gensim.models import fasttext
from gensim.test.utils import datapath

cap_path = datapath("crime-and-punishment.bin")
fb_model = fasttext.load_facebook_model(cap_path)
