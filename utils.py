from collections import Counter
from pathlib import Path
from random import random
from nltk import word_tokenize
import rouge_papier_v2
import pandas as pd
import re
import numpy as np
import os
import json 
import torch
import os
import subprocess

# Utility functions
def get_posweight(train_label_dir,file_list_file=None):
    if file_list_file != None:
        of = open(file_list_file,'r')
        file_list = of.readlines()
        of.close()
        file_l = [Path(train_label_dir+'/'+f.strip()+'.json') for f in file_list]
    else:
        label_dir = Path(train_label_dir)
        file_l = [path for path in label_dir.glob("*.json")]
    total_num=0
    total_pos = 0
    for f in file_l[:50000]:
        with f.open() as of:
            d = json.load(of)['labels']
        total_num+=len(d)
        total_pos+=sum(d)
    print('Compute pos weight done! There are %d sentences in total, with %d sentences as positive'%(total_num,total_pos))
    return torch.FloatTensor([(total_num-total_pos)/float(total_pos)])

def make_file_list(input_dir,file_list_file):
    of = open(file_list_file,'r')
    file_list = of.readlines()
    of.close()
    f_list = [Path(input_dir+'/'+f.strip()+'.json') for f in file_list]
    return f_list

def get_all_text(train_input_dir):
    if isinstance(train_input_dir,list):
        file_l = train_input_dir
    else:
        train_input = Path(train_input_dir)
        file_l = [path for path in train_input.glob("*.json")]
    all_tokens = []
    for f in file_l:
        with f.open() as of:
            d = json.load(of)
        tokens = [t for sent in d['inputs'] for t in (sent['tokens']+['<eos>'])]
        all_tokens.append(tokens)
    return all_tokens

def build_word2ind(utt_l, vocabularySize):
    word_counter = Counter([word for utt in utt_l for word in utt])
    print('%d words found!'%(len(word_counter)))
    vocabulary = ["<UNK>"] + [e[0] for e in word_counter.most_common(vocabularySize)]
    word2index = {word:index for index,word in enumerate(vocabulary)}
    global EOS_INDEX
    EOS_INDEX = word2index['<eos>']
    return word2index

# Build embedding matrix by importing the pretrained glove
def getEmbeddingMatrix(gloveDir, word2index, embedding_dim):
    '''Refer to the official baseline model provided by SemEval.'''
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with open(os.path.join(gloveDir, 'glove.6B.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector
    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(word2index) , embedding_dim))
    for word, i in word2index.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
    
    return embeddingMatrix


def get_rouge(hyp_pathlist, ref_pathlist, length_limit,remove_stopwords,stemmer,lcs=False,):
    path_data = []
    uttnames = []
    for i in range(len(hyp_pathlist)):
        path_data.append([hyp_pathlist[i], [ref_pathlist[i]]])
        uttnames.append(os.path.splitext(hyp_pathlist[i])[0].split('/')[-1])

    config_text = rouge_papier_v2.util.make_simple_config_text(path_data)
    config_path = './config'
    of = open(config_path,'w')
    of.write(config_text)
    of.close()
    uttnames.append('Average')
    df,avgfs = rouge_papier_v2.compute_rouge(
        config_path, max_ngram=2, lcs=lcs, 
        remove_stopwords=remove_stopwords,stemmer=stemmer,set_length = False, length=length_limit)
    df['data_ids'] = pd.Series(np.array(uttnames),index =df.index)
    avg = df.iloc[-1:].to_dict("records")[0]
    if lcs:
        print("Rouge-1 recall score: %f, Rouge-1 f-score: %f,\
                Rouge-2 recall score:%f, Rouge-2 f-score:%f,\
                Rouge-L recall score:%f, Rouge-L f-score:%f"%(\
                    avg['rouge-1-r'],avg['rouge-1-f'],\
                    avg['rouge-2-r'],avg['rouge-2-f'],\
                    avg['rouge-L-r'],avg['rouge-L-f']))
    else: 
        print("Rouge-1 recall score: %f, Rouge-1 f-score: %f,Rouge-2 recall score:%f, Rouge-2 f-score:%f"%(\
                    avg['rouge-1-r'],avg['rouge-1-f'],\
                    avg['rouge-2-r'],avg['rouge-2-f']))
    return avgfs[1],df
        
def get_meteor(hyp_pathlist,ref_pathlist,model_type):
    all_ref =[]
    all_hyp = []
    total_num = len(hyp_pathlist)
    for i in range(total_num):
        of = open(ref_pathlist[i],'r')
        c = of.readlines()
        c = [i.strip('\n') for i in c]
        of.close()
        all_ref.append(' '.join(c))

        of = open(hyp_pathlist[i],'r')
        c = of.readlines()
        c = [i.strip('\n') for i in c]
        of.close()
        all_hyp.append(' '.join(c))

    of = open('all_ref_inorder.txt','w')
    of.write('\n'.join(all_ref))
    of.close()


    of = open('all_hyp_inorder.txt','w')
    of.write('\n'.join(all_hyp))
    of.close()

    of = open('meteor_out_%s.txt'%(model_type),'w')
    subprocess.call(['java','-Xmx2G','-jar','meteor-1.5/meteor-1.5.jar','all_hyp_inorder.txt','all_ref_inorder.txt','-norm','-f','system1'],stdout=of)
    of.close()


