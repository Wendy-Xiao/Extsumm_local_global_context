# Extsumm_local_global_context

This is the official code for paper 'Extractive summarization of Long Documents by combining local context and global context'(EMNLP-IJCNLP 2019).

## Installation
Make sure you are using python 3.6 and pytorch installed.

First need to install the tool rouge_papier_v2. Direct to folder 'rouge_papier_v2', and then 'python setup.py install'.
(This is a modified version from https://github.com/kedz/rouge_papier)
The data should in the same form as example-input.json, example-label.json and example-abstract.txt.

## Train
To train the model, just type 
```
python main.py --train_input /path/to/input/folder/of/train/set 
				--train_label /path/to/labels/folder/of/train/set 
				--val_input /path/to/input/folder/of/val/set 
				--val_label /path/to/labels/folder/of/val/set 
				--refpath /path/to/human-abstraction/folder/of/val/set 
				--gloveDir /path/to/pretrained/embedding 
				--length_limit 200 
				--dataset arxiv 
				--device GPU_DEVICE_NUMBER 
				--model MODEL_TO_CHOOSE 
```
The options of MODEL_TO_CHOOSE is 'bsl1', 'bsl2', 'bsl3', 'concat', 'attentive_context', 'sr' and 'cl'.

### Options
**--train_input**: The path to the input folder of training set, instances should be end with '.json', and the example input file is shown in example-input.json.
**--train_label**: The path to the label folder of training set, instances should be end with '.json' and have the same name as the corresponding input. The example input file is shown in example-label.json.
**--val_input**: The path to the input folder of valiadation set, instances should be end with '.json', and the example input file is shown in example-input.json.
**--val_label**: The path to the label folder of valiadation set, instances should be end with '.json', and have the same name as the corresponding input. The example input file is shown in example-label.json.
**--ref_path**: The path to the reference folder of the valiadation, instances should be end with '.txt', and have the same name as the corresponding input. The example input file is shown in example-abstract.txt.
**--gloveDir**: The path to the file storing the pretrained glove word embeddings.
**--length_limit**: Length constraint of the generated summaries in terms of the number of words. (default=200)
**--dataset**: The dataset name, related to the vocabulary file, and the folder to store the result model. 
**--runtime**: The order of current model, used to train more than one models. Related to the folder to store the result model. (default=0)
**--num_epoch**: The maximum number of epoch. (default=50)
**--device**: GPU number, used when there is more than one GPU, and you want to use the certain GPU. (default=0)
**--model**: The model you want to train, choose from 'bsl1', 'bsl2', 'bsl3', 'concat', 'attentive_context', 'sr' and 'cl'.
**--hidden_dim**: The dimension of hidden states in RNN. (default=300)
**--mlp_size**: The dimension of final mlp layer. (default=100)
**--vocab_size**: The size of vocabulary. (default=50000)
**--cell**: Type of the RNN cell, choose one from gru, lstm (default=gru).
**--embedding_dim**: The dimension of word embedding (default=300).
**--batchsize**: The size of each mini batch (default=32).
**--remove_stopwords**: The option of Rouge, add if want to remove the stopwords.
**--stemmer**: The option of Rouge, add if want to use stemmer.
**--seed**: Set the seed of pytorch, so that you can regenerate the result. 
**--train_file_list**: If you want to train the model only on a subset of training set, list the ids in a file.
**--val_file_list**: If you want to test the model only on a subset of valiadation set, list the ids in a file.

## Test
To test the model, just type 
```
python test.py --train_input /path/to/input/folder/of/train/set 
				--train_label /path/to/labels/folder/of/train/set 
				--test_input /path/to/input/folder/of/test/set 
				--test_label /path/to/labels/folder/of/test/set 
				--refpath /path/to/human-abstraction/folder/of/test/set 
				--gloveDir /path/to/pretrained/embedding 
				--length_limit 200 
				--dataset arxiv 
				--device GPU_DEVICE_NUMBER 
				--model MODEL_TO_CHOOSE 
				--model_path path/to/pretrained/model 
```

### Options
**--train_input**: The path to the input folder of training set, instances should be end with '.json', and the example input file is shown in example-input.json. (only for computing the loss.)
**--train_label**: The path to the label folder of training set, instances should be end with '.json' and have the same name as the corresponding input. The example input file is shown in example-label.json.
**--test_input**: The path to the input folder of test set, instances should be end with '.json', and the example input file is shown in example-input.json.
**--val_label**: The path to the label folder of test set, instances should be end with '.json', and have the same name as the corresponding input. The example input file is shown in example-label.json.
**--ref_path**: The path to the reference folder of the test, instances should be end with '.txt', and have the same name as the corresponding input. The example input file is shown in example-abstract.txt.
**--gloveDir**: The path to the file storing the pretrained glove word embeddings.
**--length_limit**: Length constraint of the generated summaries in terms of the number of words. (default=200)
**--dataset**: The dataset name, related to the vocabulary file, and the folder to store the result model. 
**--runtime**: The order of current model, used to train more than one models. Related to the folder to store the result model. (default=0)
**--device**: GPU number, used when there is more than one GPU, and you want to use the certain GPU. (default=0)
**--model**: The model you want to train, choose from 'bsl1', 'bsl2', 'bsl3', 'concat', 'attentive_context', 'sr' and 'cl'.
**--hidden_dim**: The dimension of hidden states in RNN, must be the same as the pretrained model. (default=300)
**--mlp_size**: The dimension of final mlp layer, must be the same as the pretrained model. (default=100)
**--vocab_size**: The size of vocabulary, must be the same as the pretrained model. (default=50000)
**--cell**: Type of the RNN cell, choose one from gru, lstm. Must be the same as the pretrained model (default=gru).
**--embedding_dim**: The dimension of word embedding, must be the same as the pretrained model(default=300).
**--batchsize**: The size of each mini batch (default=32).
**--remove_stopwords**: The option of Rouge, add if want to remove the stopwords.
**--stemmer**: The option of Rouge, add if want to use stemmer.
**--model_path**: The path to the pretrained model.
**--train_file_list**: If you want to train the model only on a subset of training set, list the ids in a file.
**--test_file_list**: If you want to test the model only on a subset of test set, list the ids in a file.
