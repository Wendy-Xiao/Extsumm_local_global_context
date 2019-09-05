from torch import nn
from collections import Counter
from random import random
from nltk import word_tokenize
from torch.autograd import Variable
import pandas as pd
import sys
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
from timeit import default_timer as timer

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--cell", default='gru', help="Choose one from gru, lstm")
parser.add_argument("--embedding_dim", type=int, default = 300, help = "Set the dimension of word_embedding")
parser.add_argument("--hidden_dim", type=int, default = 300, help = "Set the dimension of hidden state")
parser.add_argument("--mlp_size", type=int, default = 100, help = "Set the dimension of the integrated mlp layer")
parser.add_argument("--num_epoch", type=int, default = 50, help = "Set the number of epoch")
parser.add_argument("--batchsize", type=int, default = 32, help = "Set the size of batch")
parser.add_argument("--dataset", type=str, default = 'Pubmed-full', help = "The dataset used to train")
parser.add_argument("--length_limit",type = int, default = 200, help = "length limit of extractive summarization")
parser.add_argument("--train_input",type = str, default = None, help = "The filepath of input of training data")
parser.add_argument("--train_label",type = str, default = None, help = "The filepath of label of training data")
parser.add_argument("--train_file_list",type = str, default = None, help = "The file storing the ids of a subset of training data")
parser.add_argument("--val_input",type = str, default = None, help = "The filepath of input of validation data")
parser.add_argument("--val_label",type = str, default = None, help = "The filepath of label of validation data")
parser.add_argument("--val_file_list",type = str, default = None, help = "The file storing the ids of a subset of val data")
parser.add_argument("--model",type = str, default = 'bsl1', help = "The path to save models")
parser.add_argument("--runtime", type=int, default = 0, help = "Index of this model")
parser.add_argument("--gloveDir", type=str, default = './', help = "Directory storing glove embedding")
parser.add_argument("--refpath", type=str, default = './human-abstracts/', help = "Directory storing human abstracts")
parser.add_argument("--vocab_size", type=int, default = 50000, help = "vocabulary size")
parser.add_argument("--device", type=int, default = 1, help = "device used to compute")
parser.add_argument("--remove_stopwords", action='store_true', help = "if add this flag, then set remove_stopwords to be true")
parser.add_argument("--stemmer", action='store_true', help = "if add this flag, then set stemmer to be true")
parser.add_argument("--seed", type=int, default=None, help= "Set the seed of pytorch, so that you can regenerate the result.")

args = parser.parse_args()
print(args)


# Set the global variables
HIDDEN_DIM = args.hidden_dim
NUM_EPOCH = args.num_epoch
BATCH = args.batchsize
NUM_RUN = str(args.runtime)
EMBEDDING_DIM = args.embedding_dim
MLP_SIZE = args.mlp_size
CELL_TYPE = args.cell
LENGTH_LIMIT = args.length_limit
VOCABULARY_SIZE = args.vocab_size

LEARNING_RATE = 1e-4
TEACHER_FORCING=False
USE_SECTION_INFO=False

# if seed is given, set the seed for pytorch on both cpu and gpu
if args.seed:
    torch.manual_seed(args.seed)

# reference path and the temorary path to store the generated summaries of validation set
ref_path = args.refpath
hyp_path = './eval_hyp/%s-%d/'%(args.model,args.runtime)
if not os.path.exists(hyp_path):
    os.makedirs(hyp_path)

# set the directory to store models, make new if not exists
MODEL_DIR = args.model+'-'+args.dataset+'-'+NUM_RUN
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# set the device the model running on
device = torch.device("cuda:%d"%(args.device))
torch.cuda.set_device(args.device)

# set the parameter for ROUGE, here we use the default setting
# not remove stopwords and not use stemmer
remove_stopwords = args.remove_stopwords
stemmer = args.stemmer

# set the training and validation directories
train_input_dir = args.train_input
train_label_dir = args.train_label
val_input_dir = args.val_input
val_label_dir = args.val_label
if args.train_file_list:
    train_input_dir = make_file_list(train_input_dir,args.train_file_list)
if args.val_file_list:
    val_input_dir = make_file_list(val_input_dir,args.val_file_list)

# build the vocabulary dictionary
if 'vocabulary_%s.json'%(args.dataset) in [path.name for path in Path('./').glob('*.json')]:
    with open('vocabulary_%s.json'%(args.dataset),'r') as f:
        w2v = json.load(f)
    print('Load vocabulary from vocabulary_%s.json'%(args.dataset))
else: 
    all_tokens=get_all_text(train_input_dir)
    w2v = build_word2ind(all_tokens, VOCABULARY_SIZE)
    with open('vocabulary_%s.json'%(args.dataset),'w') as f:
        json.dump(w2v,f)
sys.stdout.flush()

# get the pos weight, used in the loss function
pos_weight = get_posweight(train_label_dir,args.train_file_list)
if torch.cuda.is_available():
    pos_weight=pos_weight.to(device)

# build embedding matrix
gloveDir = args.gloveDir
embedding_matrix = getEmbeddingMatrix(gloveDir, w2v, EMBEDDING_DIM)

# set the dataset and dataloader for both training and validation set.
train_dataset = SummarizationDataset(w2v,embedding_matrix, EMBEDDING_DIM,train_input_dir,target_dir=train_label_dir)
train_dataloader = SummarizationDataLoader(train_dataset,batch_size=BATCH)

val_dataset = SummarizationDataset(w2v,embedding_matrix, EMBEDDING_DIM,val_input_dir,target_dir=val_label_dir,reference_dir = ref_path)
val_dataloader = SummarizationDataLoader(val_dataset,batch_size=BATCH)

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
    model = Attentive_context(EMBEDDING_DIM,HIDDEN_DIM, MLP_SIZE,  cell_type=CELL_TYPE)
    USE_SECTION_INFO=True
elif args.model =='cl':
    model = ChengAndLapataSentenceExtractor(EMBEDDING_DIM,HIDDEN_DIM,cell=CELL_TYPE)
    TEACHER_FORCING = True
elif args.model =='sr':
    model = SummaRunnerSentenceExtractor(EMBEDDING_DIM,HIDDEN_DIM,cell=CELL_TYPE)

if torch.cuda.is_available():
    model=model.to(device)

sys.stdout.flush()

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE,weight_decay=1e-5)
best_r2 = 0
best_ce = 1000
train_loss=[]
val_loss = []
print('Start Training!')
time_start = timer()
time_epoch_end_old = time_start
for e in range(NUM_EPOCH):
    if e==25:
        if args.model =='cl':
            model.teacher_forcing = False
            TEACHER_FORCING=False
    l = train_seq2seq(train_dataloader,model,optimizer,pos_weight,device,USE_SECTION_INFO,TEACHER_FORCING)
    train_loss.append(l)
    print('Epoch %d finished, the avg loss: %f'%(e,l))
    r2, l = eval_seq2seq(val_dataloader,model,hyp_path,LENGTH_LIMIT,pos_weight,device,USE_SECTION_INFO,remove_stopwords,stemmer)
    val_loss.append(l)
    print('Validation loss: %f'%(l))
    if r2>best_r2:
        PATH = MODEL_DIR+'/best_r2'
        best_r2 = r2
        torch.save(model.state_dict(), PATH)
        print('Epoch %d, saved as best model - highest r2.'%(e))
    if l<=best_ce:
        best_ce = l
        print('Epoch %d, lowest ce!'%(e))
    time_epoch_end_new = timer()
    print ('Seconds to execute to whole epoch: ' + str(time_epoch_end_new - time_epoch_end_old))
    time_epoch_end_old = time_epoch_end_new
    sys.stdout.flush()
print('Seconds to execute to whole training procedure: ' + str(time_epoch_end_old - time_start))




