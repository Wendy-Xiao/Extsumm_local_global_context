from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
from pathlib import Path
from torch import nn
import torch.nn.functional as F
import json
import utils
import torch
from models import *
import collections

class SummarizationDataset(Dataset):
	def __init__(self,word2index, embedding_matrix, embedding_size, input_dir,target_dir=None,
				reference_dir=None):
		self._w2i = word2index
		if isinstance(input_dir,list):
			self._inputs = input_dir
		else:
			inputs_dir = Path(input_dir)
			self._inputs = [path for path in inputs_dir.glob("*.json")]
		self._inputs.sort()
		self._target_dir = None
		self._reference_dir = None
		self.embedding_matrix = embedding_matrix

		if target_dir:
			self._target_dir = Path(target_dir)
		if reference_dir:
			self._reference_dir = reference_dir

	def __len__(self):
		return len(self._inputs)

	def __getitem__(self,idx):
		p = self._inputs[idx]
		out = {}
		with p.open() as of: 
			data = json.load(of)
		out['id'] = data['id']
		out['filename']=p
		# Document_l is a list of list of word indexes, each sublist is a sentence, and each sentence is 
		# end with a <eos>
		document_l = []
		for i in data['inputs']:
			sent_l = []
			for w in i['tokens']:
				sent_l.append(self._w2i.get(w,0))
			sent_l.append(self._w2i['<eos>'])
			sent_embed = torch.FloatTensor(self.embedding_matrix[sent_l,:])
			document_l.append(sent_embed)

		out['document'] = document_l
		out['num_sentences'] = len(out['document'])
		out['section_lengths'] = data['section_lengths']
		# If targets are given, then read the targets
		out['labels'] = None
		if self._target_dir:
			target_file = self._target_dir / "{}.json".format(out["id"])
			if target_file.exists():
				with target_file.open() as of:
					label_data = json.load(of)
				out['labels'] = label_data['labels']

		# If the reference is given, load the reference
		out['reference'] = None
		if self._reference_dir:
			ref_file = self._reference_dir +"/{}.txt".format(out["id"])
			out['reference'] = ref_file

		return out

class SummarizationDataLoader(DataLoader):
	def __init__(self,dataset, batch_size=1, shuffle=True):
		super(SummarizationDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,collate_fn =self.avgsent_batch)
	def avgsent_batch(self,batch):
		batch.sort(key=lambda x: x["num_sentences"], reverse=True)
		out = {}
		out['id'] = []
		doc_batch = []
		labels_batch = []
		doc_lengths = []
		out['refs'] = []
		out['filenames'] = []

		section_length_batch = []
		for d in batch:
			out['id'].append(d['id'])
			doc_l = torch.FloatTensor(d['num_sentences'],d['document'][0].size()[1])
			for i in range(len(d['document'])):
				doc_l[i,:] = torch.mean(d['document'][i],0)
			doc_batch.append(doc_l)
			labels_batch.append(torch.FloatTensor(d['labels']).unsqueeze(1))
			doc_lengths.append(d['num_sentences'])
			out['filenames'].append(d['filename'])
			if d['reference']!=None:
				out['refs'].append(d['reference'])

			section_length_batch.append(d['section_lengths'])

		indicators,padded_lengths = self.build_section_indicators_and_pad(section_length_batch,doc_lengths[0])
		out['indicators'] = indicators
		out['padded_lengths'] = padded_lengths

		padded_doc_batch = pad_sequence(doc_batch,padding_value=-1)
		padded_labels_batch = pad_sequence(labels_batch,padding_value=-1)
		packed_padded_doc_batch = pack_padded_sequence(padded_doc_batch,doc_lengths)
		out['document'] = packed_padded_doc_batch
		out['labels'] = padded_labels_batch
		out['input_length'] = torch.LongTensor(doc_lengths)
		return out

	def build_section_indicators_and_pad(self,section_length_batch,max_seq_length):
		max_section_num = max([len(i) for i in section_length_batch])
		batch_size = len(section_length_batch)
		# padded lengths
		padded_lengths = torch.zeros((batch_size,max_section_num),dtype=torch.int)
		# indicators
		indicators = torch.zeros((batch_size,max_seq_length,max_section_num))

		for i_sec in range(batch_size):
			section_lengths = torch.LongTensor(section_length_batch[i_sec])
			padded_lengths[i_sec,:section_lengths.shape[0]] = section_lengths
			end = torch.clamp(torch.cumsum(section_lengths,0),0,max_seq_length)
			begin = torch.cat((torch.LongTensor([0]),end[:-1]),0)
			for i in range(len(begin)):
				indicators[i_sec,begin[i]:end[i],i]=1

		return indicators,padded_lengths

