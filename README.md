# Extsumm_local_global_context

This is the official code for paper 'Extractive summarization of Long Documents by combining local context and global context'(EMNLP-IJCNLP 2019).

## Installation
Make sure you are using python 3.6 and pytorch installed.

First need to install the tool rouge_papier_v2. Direct to folder 'rouge_papier_v2', and then 'python setup.py install'.
(This is a modified version from https://github.com/kedz/rouge_papier)
The data should in the same form as example-input.json, example-label.json and example-abstract.txt.

## Train
To train the model, just type 
'''
python main.py --train_input /path/to/input/folder/of/train/set --train_label /path/to/labels/folder/of/train/set --val_input /path/to/input/folder/of/val/set --val_label /path/to/labels/folder/of/val/set --refpath /path/to/human-abstraction/folder/of/val/set --gloveDir /path/to/pretrained/embedding --length_limit 200 --dataset arxiv --num_epoch 30 --runtime 0 --device GPU_DEVICE_NUMBER --model MODEL_TO_CHOOSE --hidden_dim 300 --mlp_size 100 --vocab_size 50000
'''

## Test
To test the model, just type 
'''
python test.py --train_input /path/to/input/folder/of/train/set --train_label /path/to/labels/folder/of/train/set --test_input /path/to/input/folder/of/test/set --test_label /path/to/labels/folder/of/test/set --refpath /path/to/human-abstraction/folder/of/test/set --gloveDir /path/to/pretrained/embedding --length_limit 200 --dataset arxiv --runtime 0 --device GPU_DEVICE_NUMBER --model MODEL_TO_CHOOSE --model_path path/to/pretrained/model --hidden_dim 300 --mlp_size 100
'''

The options of MODEL_TO_CHOOSE is 'bsl1', 'concat', 'attentive_context', 'sr' and 'cl'.
