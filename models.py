from collections import Counter
from random import random
from nltk import word_tokenize
from torch import nn
from torch.autograd import Variable
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
import os


# Only use sentence representation
class Bsl1(nn.Module):
    def __init__(self,input_size,hidden_size,mlp_size, cell_type='gru'):
        super(Bsl1, self).__init__()
        self.hidden_size = hidden_size
        self.cell = cell_type
        if self.cell == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,num_layers=1, bidirectional = True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size,num_layers=1, bidirectional = True)

        self.hidden2out = self.build_mlp(hidden_size*2,mlp_size,0.3)
        self.final_layer = nn.Linear(mlp_size, 1)
        self.dropout_layer = nn.Dropout(p=0.3)

    def build_mlp(self,input,output,dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(p=dropout, inplace=True))
        return nn.Sequential(*mlp)

    def forward(self,inputs,input_length,device):
        output, hidden = self.rnn(inputs)
        output,_ = pad_packed_sequence(output)
        output = self.dropout_layer(output)
        if self.cell == 'lstm':
            hidden = hidden[0]
        # mlp_out = [seq_len,batch,hidden_size*2]
        mlp_out = self.hidden2out(output)
        # out = [seq_len, batch, 1]
        out = self.final_layer(mlp_out)
        return out

    def predict(self, score_batch, ids, input_lengths, length_limit, filenames,hyp_path):
        #score_batch = [batch,seq_len]
        correct_num = 0
        summaryfile_batch = []
        all_ids=[]
        for i in range(len(input_lengths)):
            summary = []
            selected_ids = []
            scores = score_batch[i,:(input_lengths[i])]
            sorted_linenum = [x for _,x in sorted(zip(scores,list(range(input_lengths[i]))),reverse=True)]
            fn = filenames[i]
            with fn.open() as of:
                inputs = json.load(of)['inputs']
            wc = 0
            for j in sorted_linenum:
                summary.append(inputs[j]['text'])
                selected_ids.append(j)
                wc+=inputs[j]['word_count']
                if wc>=length_limit:
                    break
            summary='\n'.join(summary)

            all_ids.append(selected_ids)
            fname = hyp_path+ids[i]+'.txt'
            of = open(fname,'w')
            of.write(summary)
            summaryfile_batch.append(fname)
        return summaryfile_batch,all_ids

# BSL1 + local
class Bsl2(nn.Module):
    def __init__(self,input_size,hidden_size, mlp_size, cell_type='gru'):
        super(Bsl2, self).__init__()
        self.hidden_size = hidden_size
        self.cell = cell_type
        if self.cell == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,num_layers=1, bidirectional = True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size,num_layers=1, bidirectional = True)
        self.hidden2out = self.build_mlp(hidden_size*4,mlp_size,0.3)
        self.final_layer = nn.Linear(mlp_size, 1)
        self.dropout_layer = nn.Dropout(p=0.3)


    def build_mlp(self,input,output,dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(p=dropout, inplace=True))
        return nn.Sequential(*mlp)

    def forward(self,inputs,input_length,section_indicator,begin,end,device):
        output, hidden = self.rnn(inputs)
        output,_ = pad_packed_sequence(output)
        if self.cell == 'lstm':
            hidden = hidden[0]
        # Hidden=[num_layers * 2, batch, hidden_size]

        hidden = hidden.permute(1,0,2)
        seq_length = output.size()[0]

        # doc_representation =[seq_len, batch, doc_size]
        local_context_representation = []
        padding = torch.zeros((1,output.shape[1],2*self.hidden_size))
        if torch.cuda.is_available():
            padding = padding.to(device)
        pad_output = torch.cat((padding,output),0)

        # output = [seq_len, batch, 2 * hidden_size]
        for i in range(output.shape[1]):
            # local_context_f = [batch,hidden_size]
            # local_context_b = [batch,hidden_size]

            local_context_f = pad_output[end[i],i,:self.hidden_size]\
                                -pad_output[begin[i],i,:self.hidden_size]
            local_context_b = pad_output[begin[i],i,self.hidden_size:]\
                                -pad_output[end[i],i,self.hidden_size:]
            local_context = torch.cat((local_context_f,local_context_b),-1)
            local_context_representation.append(local_context)
            del local_context_f,local_context_b,local_context
        # local_context_representation = [seq_len, batch, local_size]
        del pad_output,padding
        local_context_representation = torch.stack(local_context_representation,0)
        local_context_representation = torch.bmm(section_indicator,local_context_representation)
        local_context_representation = local_context_representation.permute(1,0,2)
        local_context_representation = self.dropout_layer(local_context_representation)
        output = self.dropout_layer(output)
        # mlp_in = [seq_len,batch, 6*hidden_size]
        mlp_in = torch.cat((output,local_context_representation),-1)
        del output,local_context_representation
        # mlp_out = [seq_len,batch,mlp_size]
        mlp_out = self.hidden2out(mlp_in)
        # out = [seq_len, batch, 1]
        out = self.final_layer(mlp_out)
        return out

    def predict(self, score_batch, ids, input_lengths, length_limit, filenames,hyp_path):
        #score_batch = [batch,seq_len]
        correct_num = 0
        summaryfile_batch = []
        all_ids=[]
        for i in range(len(input_lengths)):
            summary = []
            selected_ids = []
            scores = score_batch[i,:(input_lengths[i])]
            sorted_linenum = [x for _,x in sorted(zip(scores,list(range(input_lengths[i]))),reverse=True)]
            fn = filenames[i]
            with fn.open() as of:
                inputs = json.load(of)['inputs']
            wc = 0
            for j in sorted_linenum:
                summary.append(inputs[j]['text'])
                selected_ids.append(j)
                wc+=inputs[j]['word_count']
                if wc>=length_limit:
                    break
            summary='\n'.join(summary)

            all_ids.append(selected_ids)
            fname = hyp_path+ids[i]+'.txt'
            of = open(fname,'w')
            of.write(summary)
            summaryfile_batch.append(fname)
        return summaryfile_batch,all_ids
        
# BSL1 + global
class Bsl3(nn.Module):
    def __init__(self,input_size,hidden_size, mlp_size, cell_type='gru'):
        super(Bsl3, self).__init__()
        self.hidden_size = hidden_size
        self.cell = cell_type
        if self.cell == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,num_layers=1, bidirectional = True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size,num_layers=1, bidirectional = True)
        self.hidden2out = self.build_mlp(hidden_size*4,mlp_size,0.3)
        self.final_layer = nn.Linear(mlp_size, 1)
        self.dropout_layer = nn.Dropout(p=0.3)


    def build_mlp(self,input,output,dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(p=dropout, inplace=True))
        return nn.Sequential(*mlp)

    def forward(self,inputs,input_length,section_indicator,begin,end,device):
        output, hidden = self.rnn(inputs)
        output,_ = pad_packed_sequence(output)
        if self.cell == 'lstm':
            hidden = hidden[0]
        # Hidden=[num_layers * 2, batch, hidden_size]

        hidden = hidden.permute(1,0,2)
        seq_length = output.size()[0]
        doc_represent = hidden.contiguous().view(hidden.size()[0],2*hidden.size()[2])
        # doc_representation =[batch, 2*hidden_size]
        doc_represent = doc_represent.expand(seq_length,-1,-1)
        doc_represent = self.dropout_layer(doc_represent)

        # mlp_in = [seq_len,batch, 6*hidden_size]
        mlp_in = torch.cat((output,doc_represent),-1)
        del output,doc_represent
        # mlp_out = [seq_len,batch,mlp_size]
        mlp_out = self.hidden2out(mlp_in)
        # out = [seq_len, batch, 1]
        out = self.final_layer(mlp_out)
        return out

    def predict(self, score_batch, ids, input_lengths, length_limit, filenames,hyp_path):
        #score_batch = [batch,seq_len]
        correct_num = 0
        summaryfile_batch = []
        all_ids=[]
        for i in range(len(input_lengths)):
            summary = []
            selected_ids = []
            scores = score_batch[i,:(input_lengths[i])]
            sorted_linenum = [x for _,x in sorted(zip(scores,list(range(input_lengths[i]))),reverse=True)]
            fn = filenames[i]
            with fn.open() as of:
                inputs = json.load(of)['inputs']
            wc = 0
            for j in sorted_linenum:
                summary.append(inputs[j]['text'])
                selected_ids.append(j)
                wc+=inputs[j]['word_count']
                if wc>=length_limit:
                    break
            summary='\n'.join(summary)

            all_ids.append(selected_ids)
            fname = hyp_path+ids[i]+'.txt'
            of = open(fname,'w')
            of.write(summary)
            summaryfile_batch.append(fname)
        return summaryfile_batch,all_ids

# Concatenation decoder
class Concatenation(nn.Module):
    def __init__(self,input_size,hidden_size, mlp_size, cell_type='gru'):
        super(Concatenation, self).__init__()
        self.hidden_size = hidden_size
        self.cell = cell_type
        if self.cell == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,num_layers=1, bidirectional = True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size,num_layers=1, bidirectional = True)
        self.hidden2out = self.build_mlp(hidden_size*6,mlp_size,0.3)
        self.final_layer = nn.Linear(mlp_size, 1)
        self.dropout_layer = nn.Dropout(p=0.3)


    def build_mlp(self,input,output,dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(p=dropout, inplace=True))
        return nn.Sequential(*mlp)

    def forward(self,inputs,input_length,section_indicator,begin,end,device):
        output, hidden = self.rnn(inputs)
        output,_ = pad_packed_sequence(output)
        if self.cell == 'lstm':
            hidden = hidden[0]
        # Hidden=[num_layers * 2, batch, hidden_size]

        hidden = hidden.permute(1,0,2)
        seq_length = output.size()[0]
        doc_represent = hidden.contiguous().view(hidden.size()[0],2*hidden.size()[2])
        # doc_representation =[batch, 2*hidden_size]

        doc_represent = doc_represent.expand(seq_length,-1,-1)
        doc_represent = self.dropout_layer(doc_represent)
        # doc_representation =[seq_len, batch, doc_size]
        local_context_representation = []
        padding = torch.zeros((1,output.shape[1],2*self.hidden_size))
        if torch.cuda.is_available():
            padding = padding.to(device)
        pad_output = torch.cat((padding,output),0)

        # output = [seq_len, batch, 2 * hidden_size]
        for i in range(output.shape[1]):
            # local_context_f = [batch,hidden_size]
            # local_context_b = [batch,hidden_size]

            local_context_f = pad_output[end[i],i,:self.hidden_size]\
                                -pad_output[begin[i],i,:self.hidden_size]
            local_context_b = pad_output[begin[i],i,self.hidden_size:]\
                                -pad_output[end[i],i,self.hidden_size:]
            local_context = torch.cat((local_context_f,local_context_b),-1)
            local_context_representation.append(local_context)
            del local_context_f,local_context_b,local_context
        # local_context_representation = [ batch, seq_len,local_size]
        del pad_output,padding
        local_context_representation = torch.stack(local_context_representation,0)
        local_context_representation = torch.bmm(section_indicator,local_context_representation)
        local_context_representation = local_context_representation.permute(1,0,2)
        local_context_representation = self.dropout_layer(local_context_representation)
        output = self.dropout_layer(output)
        # mlp_in = [seq_len,batch, 6*hidden_size]
        mlp_in = torch.cat((output,local_context_representation,doc_represent),-1)
        del output,local_context_representation,doc_represent
        # mlp_out = [seq_len,batch,mlp_size]
        mlp_out = self.hidden2out(mlp_in)
        # out = [seq_len, batch, 1]
        out = self.final_layer(mlp_out)
        return out

    def predict(self, score_batch, ids, input_lengths, length_limit, filenames,hyp_path):
        #score_batch = [batch,seq_len]
        correct_num = 0
        summaryfile_batch = []
        all_ids=[]
        for i in range(len(input_lengths)):
            summary = []
            selected_ids = []
            scores = score_batch[i,:(input_lengths[i])]
            sorted_linenum = [x for _,x in sorted(zip(scores,list(range(input_lengths[i]))),reverse=True)]
            fn = filenames[i]
            with fn.open() as of:
                inputs = json.load(of)['inputs']
            wc = 0
            for j in sorted_linenum:
                summary.append(inputs[j]['text'])
                selected_ids.append(j)
                wc+=inputs[j]['word_count']
                if wc>=length_limit:
                    break
            summary='\n'.join(summary)

            all_ids.append(selected_ids)
            fname = hyp_path+ids[i]+'.txt'
            of = open(fname,'w')
            of.write(summary)
            summaryfile_batch.append(fname)
        return summaryfile_batch,all_ids

# # Attentive context decoder
class Attentive_context(nn.Module):
    def __init__(self,input_size,hidden_size, mlp_size, cell_type='gru'):
        super(Attentive_context, self).__init__()
        self.hidden_size = hidden_size
        self.cell = cell_type
        if self.cell == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size,num_layers=1, bidirectional = True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size,num_layers=1, bidirectional = True)

        self.hidden2out = self.build_mlp(hidden_size*4,mlp_size,0.3)
        self.final_layer = nn.Linear(mlp_size, 1)
        self.feat_attn = nn.Linear(hidden_size*4,hidden_size*4,bias=False)
        self.context_vector = nn.Parameter(torch.rand(hidden_size*4,1))
        self.dropout_layer = nn.Dropout(p=0.3)


    def build_mlp(self,input,output,dropout):
        mlp = []
        mlp.append(nn.Linear(input, output))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(p=dropout, inplace=True))
        return nn.Sequential(*mlp)

    def attention_net(self, features, sent_representation):
        # features_tanh = torch.tanh(features)
        sent_representation = sent_representation.unsqueeze(1)
        sent_representation = sent_representation.expand(-1,3,-1) # [batch,3,hidden*2]
        f = torch.cat([sent_representation,features],-1) #(batch,3,hidden*4)
        context = torch.tanh(self.feat_attn(f)) # (batch,3,hidden*4)
        v = self.context_vector.unsqueeze(0).expand(context.size()[0],-1,-1)
        attn_weights = torch.bmm(context,v)

        soft_attn_weights = F.softmax(attn_weights, 1) #(batch,3,1)

        new_hidden_state = torch.bmm(features.transpose(1, 2), soft_attn_weights).squeeze(2) #[batch,hidden*2]
        return new_hidden_state

    def forward(self,inputs,input_length,section_indicator,begin,end,device):
        output, hidden = self.rnn(inputs)
        output,_ = pad_packed_sequence(output)
        if self.cell == 'lstm':
            hidden = hidden[0]
        # Hidden=[num_layers * 2, batch, hidden_size]
        # hidden = self.dropout_layer(hidden)
        hidden = hidden.permute(1,0,2)
        seq_length = output.size()[0]
        doc_represent = hidden.contiguous().view(hidden.size()[0],2*hidden.size()[2])
        # doc_representation =[batch, 2*hidden_size]

        doc_represent = doc_represent.expand(seq_length,-1,-1)
        doc_represent = self.dropout_layer(doc_represent)
        # doc_representation =[seq_len, batch, doc_size]
        local_context_representation = []
        padding = torch.zeros((1,output.shape[1],2*self.hidden_size))
        if torch.cuda.is_available():
            padding = padding.to(device)
        pad_output = torch.cat((padding,output),0)

        # output = [seq_len, batch, 2 * hidden_size]
        for i in range(output.shape[1]):
            # local_context_f = [batch,hidden_size]
            # local_context_b = [batch,hidden_size]

            local_context_f = pad_output[end[i],i,:self.hidden_size]\
                                -pad_output[begin[i],i,:self.hidden_size]
            local_context_b = pad_output[begin[i],i,self.hidden_size:]\
                                -pad_output[end[i],i,self.hidden_size:]
            local_context = torch.cat((local_context_f,local_context_b),-1)
            local_context_representation.append(local_context)
            del local_context_f,local_context_b,local_context
        # local_context_representation = [seq_len, batch, local_size]
        del pad_output,padding
        local_context_representation = torch.stack(local_context_representation,0)
        local_context_representation = torch.bmm(section_indicator,local_context_representation)
        local_context_representation = local_context_representation.permute(1,0,2)
        local_context_representation = self.dropout_layer(local_context_representation)
        output = self.dropout_layer(output)
        context = torch.stack((local_context_representation,doc_represent),2)
        attn_in = torch.cat((context,output.unsqueeze(2).expand(-1,-1,2,-1)),3)
        attn_in = torch.tanh(attn_in)
        attn_weight = F.softmax(torch.matmul(self.feat_attn(attn_in),self.context_vector),dim=2)
        
        context = context.permute(0,1,3,2)
        weighted_context_representation = torch.matmul(context,attn_weight).squeeze(-1)

        mlp_in = torch.cat([weighted_context_representation,output],-1)

        # mlp_out = [seq_len,batch,mlp_size]
        mlp_out = self.hidden2out(mlp_in)
        # out = [seq_len, batch, 1]
        out = self.final_layer(mlp_out)
        return out

    def predict(self, score_batch, ids, input_lengths, length_limit, filenames,hyp_path):
        #score_batch = [batch,seq_len]
        correct_num = 0
        summaryfile_batch = []
        all_ids=[]
        for i in range(len(input_lengths)):
            summary = []
            selected_ids = []
            scores = score_batch[i,:(input_lengths[i])]
            sorted_linenum = [x for _,x in sorted(zip(scores,list(range(input_lengths[i]))),reverse=True)]
            fn = filenames[i]
            with fn.open() as of:
                inputs = json.load(of)['inputs']
            wc = 0
            for j in sorted_linenum:
                summary.append(inputs[j]['text'])
                selected_ids.append(j)
                wc+=inputs[j]['word_count']
                if wc>=length_limit:
                    break
            summary='\n'.join(summary)

            all_ids.append(selected_ids)
            fname = hyp_path+ids[i]+'.txt'
            of = open(fname,'w')
            of.write(summary)
            summaryfile_batch.append(fname)
        return summaryfile_batch,all_ids


# Cheng and Lapata, borrowed from https://github.com/kedz/nnsum/tree/emnlp18-release
class ChengAndLapataSentenceExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, cell="gru",
                 rnn_dropout=0.0, mlp_layers=[100], mlp_dropouts=[.25]):

        super(ChengAndLapataSentenceExtractor, self).__init__()
        if cell not in ["gru", "lstm", "rnn"]:
            raise Exception(("cell expected one of 'gru', 'lstm', or 'rnn' "
                             "but got {}").format(cell))

        if cell == "gru":
            self.encoder_rnn = nn.GRU(
                input_size, hidden_size, num_layers=num_layers, 
                dropout=rnn_dropout if num_layers > 1 else 0., 
                bidirectional=False)
            self.decoder_rnn = nn.GRU(
                input_size, hidden_size, num_layers=num_layers, 
                dropout=rnn_dropout if num_layers > 1 else 0.,
                bidirectional=False)
        elif cell == "lstm":
            self.encoder_rnn = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout if num_layers > 1 else 0.,
                bidirectional=False)
            self.decoder_rnn = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout if num_layers > 1 else 0.,
                bidirectional=False)
        else:
            self.encoder_rnn = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout if num_layers > 1 else 0.,
                bidirectional=False)
            self.decoder_rnn = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                dropout=rnn_dropout if num_layers > 1 else 0.,
                bidirectional=False)

        self.decoder_start = nn.Parameter(
            torch.FloatTensor(input_size).normal_())

        self.rnn_dropout = rnn_dropout

        self.teacher_forcing = True

        inp_size = hidden_size * 2
        mlp = []
        for out_size, dropout in zip(mlp_layers, mlp_dropouts):
            mlp.append(nn.Linear(inp_size, out_size))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(p=dropout))
            inp_size = out_size 
        mlp.append(nn.Linear(inp_size, 1))
        self.mlp = nn.Sequential(*mlp)

    def _apply_rnn(self, rnn, packed_input, rnn_state=None, batch_first=True):
        packed_output, updated_rnn_state = rnn(packed_input, rnn_state)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, 
            batch_first=batch_first)
        output = F.dropout(output, p=self.rnn_dropout, training=self.training)
        return output, updated_rnn_state

    def _teacher_forcing_forward(self, packed_sentence_embeddings, length_list, 
                                 targets):
        sentence_embeddings,_ = torch.nn.utils.rnn.pad_packed_sequence(packed_sentence_embeddings)

        sentence_embeddings = sentence_embeddings.permute(1,0,2)
        batch_size = sentence_embeddings.size(0)

        packed_sentence_embeddings = nn.utils.rnn.pack_padded_sequence(
            sentence_embeddings, length_list, batch_first=True)
        # [b,seq_l,embed]
        encoder_output, rnn_state = self._apply_rnn(
            self.encoder_rnn, 
            packed_sentence_embeddings)
        
        weighted_decoder_input = sentence_embeddings[:,:-1] \
            * targets.view(batch_size, -1,1)[:,:-1]

        start_emb = self.decoder_start.view(1, 1, -1).repeat(batch_size, 1, 1)
        decoder_input = torch.cat([start_emb, weighted_decoder_input], 1)

        packed_decoder_input = nn.utils.rnn.pack_padded_sequence(
                decoder_input, length_list, batch_first=True)
        decoder_output, _ = self._apply_rnn(
            self.decoder_rnn, packed_decoder_input, rnn_state=rnn_state)
        mlp_input = torch.cat([encoder_output, decoder_output], 2)
        logits = self.mlp(mlp_input).transpose(1, 0)
        return logits

    def _predict_forward(self, packed_sentence_embeddings, length_list):

        sentence_embeddings,_ = torch.nn.utils.rnn.pad_packed_sequence(packed_sentence_embeddings)


        sequence_size = sentence_embeddings.size(0)
        batch_size = sentence_embeddings.size(1)

        packed_sentence_embeddings = nn.utils.rnn.pack_padded_sequence(
            sentence_embeddings, 
            length_list, batch_first=False)

        encoder_output, rnn_state = self._apply_rnn(
            self.encoder_rnn, 
            packed_sentence_embeddings, 
            batch_first=False)
 
        encoder_outputs = encoder_output.split(1, dim=0)
        del encoder_output
        start_emb = self.decoder_start.view(1, 1, -1).repeat(1, batch_size, 1)
        decoder_inputs = sentence_embeddings.split(1, dim=0)

        logits = []
        decoder_input_t = start_emb
        for t in range(sequence_size):
            decoder_output_t, rnn_state = self.decoder_rnn(
                decoder_input_t, rnn_state)
            decoder_output_t = F.dropout(
                decoder_output_t, p=self.rnn_dropout, training=self.training)
            mlp_input_t = torch.cat([encoder_outputs[t], decoder_output_t], 2)
            logits_t = self.mlp(mlp_input_t)
            logits.append(logits_t)

            if t + 1 != sequence_size:
                probs_t = torch.sigmoid(logits_t)
                decoder_input_t = decoder_inputs[t] * probs_t

        logits = torch.cat(logits, 0)


        return logits

    def forward(self, sentence_embeddings, num_sentences, device,targets=None):
        if self.training and self.teacher_forcing:
            return self._teacher_forcing_forward(
                sentence_embeddings, num_sentences, targets)
        else:
            return self._predict_forward(sentence_embeddings, num_sentences)

    def predict(self, score_batch, ids, input_lengths, length_limit, filenames,hyp_path):
        #score_batch = [batch,seq_len]
        correct_num = 0
        summaryfile_batch = []
        all_ids=[]
        for i in range(len(input_lengths)):
            summary = []
            selected_ids = []
            scores = score_batch[i,:(input_lengths[i])]
            sorted_linenum = [x for _,x in sorted(zip(scores,list(range(input_lengths[i]))),reverse=True)]
            fn = filenames[i]
            with fn.open() as of:
                inputs = json.load(of)['inputs']
            wc = 0
            for j in sorted_linenum:
                summary.append(inputs[j]['text'])
                selected_ids.append(j)
                wc+=inputs[j]['word_count']
                if wc>=length_limit:
                    break
            summary='\n'.join(summary)

            all_ids.append(selected_ids)
            fname = hyp_path+ids[i]+'.txt'
            of = open(fname,'w')
            of.write(summary)
            summaryfile_batch.append(fname)
        return summaryfile_batch,all_ids

# SummaRunner, borrowed from https://github.com/kedz/nnsum/tree/emnlp18-release
class SummaRunnerSentenceExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1, 
                 bidirectional=True, cell="gru", rnn_dropout=0.0,
                 sentence_size=100, document_size=200,
                 segments=4, max_position_weights=25,
                 segment_size=50, position_size=50):

        super(SummaRunnerSentenceExtractor, self).__init__()

        if cell not in ["gru", "lstm", "rnn"]:
            raise Exception(("cell expected one of 'gru', 'lstm', or 'rnn' "
                             "but got {}").format(cell))
        if cell == "gru":
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers=num_layers, 
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.) 
        elif cell == "lstm":
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.)
        else:
            self.rnn = nn.RNN(
                input_size, hidden_size, num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout if num_layers > 1 else 0.)

        self.rnn_dropout = rnn_dropout

        self.teacher_forcing = True

        inp_size = hidden_size
        if bidirectional:
            inp_size *= 2

        self.sentence_rep = nn.Sequential(
            nn.Linear(inp_size, sentence_size), nn.ReLU())
        self.content_logits = nn.Linear(sentence_size, 1)
        self.document_rep = nn.Sequential(
            nn.Linear(sentence_size, document_size), 
            nn.Tanh(), 
            nn.Linear(document_size, sentence_size))
        self.similarity = nn.Bilinear(
            sentence_size, sentence_size, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor([0]))

        self.max_position_weights = max_position_weights
        self.segments = segments
        self.position_encoder = nn.Sequential(
            nn.Embedding(max_position_weights + 1, position_size, 
                         padding_idx=0),
            nn.Linear(position_size, 1, bias=False))
        self.segment_encoder = nn.Sequential(
            nn.Embedding(segments + 1, segment_size, padding_idx=0),
            nn.Linear(segment_size, 1, bias=False))


    def novelty(self, sentence_state, summary_rep):
        sim = self.similarity(
            sentence_state.squeeze(1), torch.tanh(summary_rep).squeeze(1))
        novelty = -sim.squeeze(1)
        return novelty

    def position_logits(self, length):
        batch_size = length.size(0)
        abs_pos = torch.arange(
            1, length.data[0].item() + 1, device=length.device)\
            .view(1, -1).repeat(batch_size, 1)

        chunk_size = (length.float() / self.segments).round().view(-1, 1)
        rel_pos = (abs_pos.float() / chunk_size).ceil().clamp(
            0, self.segments).long()

        abs_pos.data.clamp_(0, self.max_position_weights)
        pos_logits = self.position_encoder(abs_pos).squeeze(2)
        seg_logits = self.segment_encoder(rel_pos).squeeze(2)
        return pos_logits, seg_logits

    def forward(self, packed_sentence_embeddings, num_sentences, targets=None):

        sentence_embeddings,_ = torch.nn.utils.rnn.pad_packed_sequence(packed_sentence_embeddings)

        sentence_embeddings = sentence_embeddings.permute(1,0,2)

        packed_sentence_embeddings = nn.utils.rnn.pack_padded_sequence(
            sentence_embeddings, 
            num_sentences.data.tolist(), 
            batch_first=True)
        del sentence_embeddings
        rnn_output_packed, _ = self.rnn(packed_sentence_embeddings)
        del packed_sentence_embeddings
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
            rnn_output_packed, 
            batch_first=True)

        rnn_output = F.dropout(rnn_output, p=self.rnn_dropout, inplace=True,
                               training=self.training)

        sentence_states = self.sentence_rep(rnn_output)
        del rnn_output
        content_logits = self.content_logits(sentence_states).squeeze(2)

        avg_sentence = sentence_states.sum(1).div_(
            num_sentences.view(-1, 1).float())
        doc_rep = self.document_rep(avg_sentence).unsqueeze(2)
        del avg_sentence
        salience_logits = sentence_states.bmm(doc_rep).squeeze(2)
        del doc_rep
        pos_logits, seg_logits = self.position_logits(num_sentences)

        static_logits = content_logits + salience_logits + pos_logits \
            + seg_logits + self.bias.unsqueeze(0)
        
        sentence_states = sentence_states.split(1, dim=1)
        summary_rep = torch.zeros_like(sentence_states[0])
        logits = []
        for step in range(num_sentences[0].item()):
            novelty_logits = self.novelty(sentence_states[step], summary_rep)
            logits_step = static_logits[:, step] + novelty_logits
            del novelty_logits
            prob = torch.sigmoid(logits_step)
            
            summary_rep += sentence_states[step] * prob.view(-1, 1, 1)
            logits.append(logits_step.view(-1, 1))
        logits = torch.cat(logits, 1).unsqueeze(-1)
        logits = logits.permute(1,0,2)
        return logits
    def predict(self, score_batch, ids, input_lengths, length_limit, filenames,hyp_path):
        #score_batch = [batch,seq_len]

        correct_num = 0
        summaryfile_batch = []
        all_ids = []
        for i in range(len(input_lengths)):
            summary = []
 
            scores = score_batch[i,:(input_lengths[i])]
            sorted_linenum = [x for _,x in sorted(zip(scores,list(range(input_lengths[i]))),reverse=True)]
            fn = filenames[i]
            selected_ids = []
            with fn.open() as of:
                inputs = json.load(of)['inputs']
            wc = 0
            for j in sorted_linenum:
                summary.append(inputs[j]['text'])
                selected_ids.append(j)
                wc+=inputs[j]['word_count']
                if wc>=length_limit:
                    break
            summary='\n'.join(summary)

            all_ids.append(selected_ids)
            fname = hyp_path+ids[i]+'.txt'
            of = open(fname,'w')
            of.write(summary)
            summaryfile_batch.append(fname)
        return summaryfile_batch,all_ids