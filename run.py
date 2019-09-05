from collections import Counter
from random import random
from nltk import word_tokenize
from torch import nn
from torch.autograd import Variable
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from utils import *



def train_seq2seq(train_dataloader,model,optimizer,pos_weight,device,use_section_info,teacher_forcing):
	model.train()
	total_loss=0
	total_data=0
	for i,data in enumerate(train_dataloader):
		l,num_data = train_seq2seq_batch(data, model, optimizer,pos_weight,device,use_section_info,teacher_forcing)
		total_loss+=l
		total_data+=num_data
		if i%200==0:
			print('Batch %d, Loss: %f'%(i,total_loss/float(total_data)))
	return total_loss/float(total_data)


def train_seq2seq_batch(data_batch, model, optimizer,pos_weight,device,use_section_info=False,teacher_forcing=False):
	document = data_batch['document']
	label = data_batch['labels']
	input_length = data_batch['input_length']
	indicators = data_batch['indicators']
	padded_lengths = data_batch['padded_lengths']

	# Section information
	total_data = torch.sum(input_length)
	end = torch.clamp(torch.cumsum(padded_lengths,1),0,input_length[0])
	begin = torch.cat((torch.zeros((len(input_length),1),dtype=torch.long),end[:,:-1]),1)

	if torch.cuda.is_available():
		document = document.to(device)
		label = label.to(device)
		input_length = input_length.to(device)
		indicators = indicators.to(device)
		end = end.to(device)
		begin= begin.to(device)

	if use_section_info:
		out = model(document,input_length,indicators,begin,end,device)
	elif teacher_forcing:
		out = model(document,input_length,device,targets=label)
	else:
		out = model(document,input_length,device)

	mask = label.gt(-1).float()
	loss = F.binary_cross_entropy_with_logits(out,label,weight = mask,reduction='sum',pos_weight=pos_weight)
	model.zero_grad()
	loss.backward()
	optimizer.step()
	l = loss.data
	del document,label,input_length,indicators,end,begin,loss,out
	torch.cuda.empty_cache()
	return l,total_data


def eval_seq2seq(val_dataloader,model,hyp_path,length_limit,pos_weight,device,use_section_info ,remove_stopwords,stemmer,meteor=False,lcs=False,saveas=None):
	model.eval()
	total_loss=0
	total_data=0
	summ_path = []
	ref_path = []
	total_correct = 0
	all_ids = []
	all_oracle = []
	all_sections=[]
	sigmoid = torch.nn.Sigmoid()
	for i,data in enumerate(val_dataloader):
		summaryfiles,referencefiles,loss,num_data,select_ids,oracle,sections = eval_seq2seq_batch(sigmoid,data, model,hyp_path,length_limit,pos_weight,device,use_section_info)		
		summ_path.extend(summaryfiles)
		ref_path.extend(referencefiles)
		all_ids.extend(select_ids)
		all_sections.extend(sections)
		all_oracle.extend(oracle)
		total_loss+=loss
		total_data+=num_data
		del data
		del loss

		if i%2000==1:
			print('Batch %d, Loss: %f'%(i,total_loss/float(total_data)))

	rouge2,df = get_rouge(summ_path, ref_path, length_limit,remove_stopwords,stemmer,lcs)
	if meteor:
		model_type = type(model).__name__
		get_meteor(summ_path, ref_path,model_type)
	if saveas:
		all_sections.append([0])
		all_oracle.append([0])
		all_ids.append([0])
		df['selected'] = pd.Series(np.array(all_ids),index =df.index)
		df['oracle'] = pd.Series(np.array(all_oracle),index =df.index)
		df['sections'] = pd.Series(np.array(all_sections),index =df.index)
		df.to_csv('%s.csv'%(saveas))
	return rouge2, total_loss/float(total_data)


def eval_seq2seq_batch(sigmoid,data_batch,model,hyp_path,length_limit,pos_weight,device,use_section_info):
	document = data_batch['document']
	label = data_batch['labels']
	input_length = data_batch['input_length']
	indicators = data_batch['indicators']
	padded_lengths = data_batch['padded_lengths']

	total_data = torch.sum(input_length)
	end = torch.clamp(torch.cumsum(padded_lengths,1),0,input_length[0])
	begin = torch.cat((torch.zeros((len(input_length),1),dtype=torch.long),end[:,:-1]),1)
	if torch.cuda.is_available():
		document = document.to(device)
		label = label.to(device)
		input_length = input_length.to(device)
		indicators = indicators.to(device)
		end = end.to(device)
		begin= begin.to(device)


	reference = data_batch['refs']

	filenames = data_batch['filenames']
	ids = data_batch['id']

	if use_section_info:
		out= model(document,input_length,indicators,begin,end,device)
	else:
		out = model(document,input_length,device)

	mask = label.gt(-1).float()
	loss = F.binary_cross_entropy_with_logits(out,label,weight = mask,reduction='sum',pos_weight=pos_weight)
	out = out.squeeze(-1)
	scores = sigmoid(out).data
	scores = scores.permute(1,0)
	np.save('scores',scores.cpu().data.numpy())
	summaryfiles,all_ids= model.predict(scores, ids, input_length, length_limit, filenames,hyp_path)

	label = label.squeeze(-1)
	label = label.permute(1,0)

	all_oracle = [list((label[i]==1).nonzero().squeeze(-1).cpu().numpy()) for i in range(label.shape[0])]
	sections = [list(torch.unique(end[i],sorted=True).cpu().numpy()) for i in range(end.shape[0])]
	del document,label,input_length,indicators,end,begin
	return summaryfiles,reference,loss.data,total_data,all_ids,all_oracle,sections



