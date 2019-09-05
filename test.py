from torch import nn
from collections import Counter
from random import random
from nltk import word_tokenize
from torch.autograd import Variable
import pandas as pd
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from sklearn.utils import shuffle
import json 
import random
import argparse
from data import *
from utils import *
from run import *
from models import *
import sys


# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cell", default='gru', help="Choose one from gru, lstm")
parser.add_argument("--embedding_dim", type=int, default = 300, help = "Set the dimension of word_embedding")
parser.add_argument("--hidden_dim", type=int, default = 300, help = "Set the dimension of hidden state")
parser.add_argument("--mlp_size", type=int, default = 100, help = "Set the dimension of the integrated mlp layer")
parser.add_argument("--batchsize", type=int, default = 32, help = "Set the size of batch")
parser.add_argument("--dataset", type=str, default = 'Pubmed', help = "The dataset used to train")
parser.add_argument("--length_limit",type = int, default = 290, help = "length limit of extractive summarization")
parser.add_argument("--train_input",type = str, default = None, help = "The filepath of input of training data")
parser.add_argument("--train_label",type = str, default = None, help = "The filepath of label of training data")
parser.add_argument("--train_file_list",type = str, default = None, help = "The file storing the ids of a subset of training data")
parser.add_argument("--test_input",type = str, default = None, help = "The filepath of input of validation data")
parser.add_argument("--test_label",type = str, default = None, help = "The filepath of label of validation data")
parser.add_argument("--test_file_list",type = str, default = None, help = "The file storing the ids of a subset of test data")
parser.add_argument("--model",type = str, default = 'bsl1', help = "The path to save models")
parser.add_argument("--runtime", type=int, default = 0, help = "Index of this model")
parser.add_argument("--gloveDir", type=str, default = './', help = "Directory storing glove embedding")
parser.add_argument("--refpath", type=str, default = './human-abstracts/', help = "Directory storing human abstracts")
parser.add_argument("--model_path",type=str, default='./pretrained_models/Pubmed/concat',help="The path of model")
parser.add_argument("--device", type=int, default = 0, help = "device used to compute")
parser.add_argument("--remove_stopwords", action='store_true', help = "if add this flag, then set remove_stopwords to be true")
parser.add_argument("--stemmer", action='store_true', help = "if add this flag, then set stemmer to be true")
parser.add_argument("--result_file_name",type = str, default = None, help = "The file storing all the rouge results of test data")

args = parser.parse_args()
print(args)

# Set the global variables
HIDDEN_DIM = args.hidden_dim
BATCH = args.batchsize
NUM_RUN = str(args.runtime)
EMBEDDING_DIM = args.embedding_dim
MLP_SIZE = args.mlp_size
CELL_TYPE = args.cell
LENGTH_LIMIT = args.length_limit
LEARNING_RATE = 1e-4
USE_SECTION_INFO=False
MODEL_PATH = args.model_path
SAVE_RESULT_NAME = args.result_file_name

# Set the refpath (human-abstraction) and hyp-path(to store the generated summary)
ref_path = args.refpath
hyp_path = './test_hyp/%s-%s-%d/'%(args.model,args.dataset,args.runtime)
if not os.path.exists(hyp_path):
    os.makedirs(hyp_path)

device = torch.device("cuda:%d"%(args.device))
torch.cuda.set_device(args.device)
# Set the ROUGE parameter
remove_stopwords = args.remove_stopwords
stemmer = args.stemmer


# If the test is on a subset of the whole dataset
train_input_dir = args.train_input
train_label_dir = args.train_label
test_input_dir = args.test_input
test_label_dir = args.test_label

if args.train_file_list:
    train_input_dir = make_file_list(train_input_dir,args.train_file_list)
if args.test_file_list:
    test_input_dir = make_file_list(test_input_dir,args.test_file_list)

if 'vocabulary_%s.json'%(args.dataset) in [path.name for path in Path('./').glob('*.json')]:
    with open('vocabulary_%s.json'%(args.dataset),'r') as f:
        w2v = json.load(f)
    print('Load vocabulary from vocabulary_%s.json'%(args.dataset))
else: 
    all_tokens=get_all_text(train_input_dir)
    w2v = build_word2ind(all_tokens, VOCABULARY_SIZE)
    with open('vocabulary_%s.json'%(args.dataset),'w') as f:
        json.dump(w2v,f)

# Get the postive weight to compute the weighted loss
pos_weight = get_posweight(train_label_dir,args.train_file_list)
if torch.cuda.is_available():
    pos_weight=pos_weight.to(device)

# Load pretrained word embedding
gloveDir = args.gloveDir
embedding_matrix = getEmbeddingMatrix(gloveDir, w2v, EMBEDDING_DIM)

# Build the dataset
test_dataset = SummarizationDataset(w2v,  embedding_matrix, EMBEDDING_DIM,test_input_dir,target_dir=test_label_dir,reference_dir = ref_path)
test_dataloader = SummarizationDataLoader(test_dataset,batch_size=BATCH,shuffle=False)


print('Start loading model.')
# Initialize the model
if args.model =='bsl1':
    model = Bsl1(EMBEDDING_DIM,HIDDEN_DIM, MLP_SIZE, cell_type=CELL_TYPE)
elif args.model =='bsl2':
    model = Bsl2(EMBEDDING_DIM,HIDDEN_DIM, MLP_SIZE, cell_type=CELL_TYPE)
    USE_SECTION_INFO=True
elif args.model =='bsl3':
    model = Bsl3(EMBEDDING_DIM,HIDDEN_DIM, MLP_SIZE, cell_type=CELL_TYPE)
    USE_SECTION_INFO=True
elif args.model =='concat':
    model = Concatenation(EMBEDDING_DIM,HIDDEN_DIM, MLP_SIZE, cell_type=CELL_TYPE)
    USE_SECTION_INFO=True
elif args.model =='attentive_context':
    model = Attentive_context(EMBEDDING_DIM,HIDDEN_DIM, MLP_SIZE, cell_type=CELL_TYPE)
    USE_SECTION_INFO=True
elif args.model =='cl':
    model = ChengAndLapataSentenceExtractor(EMBEDDING_DIM,HIDDEN_DIM)
elif args.model =='sr':
    model = SummaRunnerSentenceExtractor(EMBEDDING_DIM,HIDDEN_DIM)

# Load the pre-trained model
model.load_state_dict(torch.load(MODEL_PATH))

# Move to GPU
if torch.cuda.is_available():
    model=model.to(device)

# We also want the Rouge-L score
lcs= True
model.eval()
print('Start evaluating.')
r2, l = eval_seq2seq(test_dataloader,model,hyp_path,LENGTH_LIMIT,pos_weight,device,USE_SECTION_INFO,remove_stopwords,stemmer,meteor=True,lcs=lcs,saveas=SAVE_RESULT_NAME)
print('test loss: %f'%(l))
sys.stdout.flush()