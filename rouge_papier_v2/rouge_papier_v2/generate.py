from .util import TempFileManager, make_simple_config_text
from . import wrapper
import numpy as np

def compute_extract(sentences, summaries, mode="independent", ngram=1, 
                    length=100, length_unit="word", remove_stopwords=False):

    if mode == "independent":
        return compute_greedy_independent_extract(
            sentences, summaries, ngram, length=length, 
            length_unit=length_unit, remove_stopwords=remove_stopwords)
    elif mode == "sequential":
        return compute_greedy_sequential_extract(
            sentences, summaries, ngram, length=length, 
            length_unit=length_unit, remove_stopwords=remove_stopwords), None
    else:
        raise Exception("mode must be 'independent' or 'sequential'")

def compute_pairwise_ranks(sentences, summaries, mode="independent", ngram=1, 
                           length=100, length_unit="word", 
                           remove_stopwords=False):

    if mode == "independent":
        return compute_greedy_independent_pairwise_ranks(
            sentences, summaries, ngram, length=length, 
            length_unit=length_unit, remove_stopwords=remove_stopwords)
    elif mode == "sequential":
        return compute_greedy_sequential_pairwise_ranks(
            sentences, summaries, ngram, length=length, 
            length_unit=length_unit, remove_stopwords=remove_stopwords)
    else:
        raise Exception("mode must be 'independent' or 'sequential'")

def compute_greedy_independent_extract(sentences, summaries, order, 
                                       length=100, length_unit="word",
                                       remove_stopwords=False):
    
    with TempFileManager() as manager:
        input_paths = manager.create_temp_files(sentences)
        summary_paths = manager.create_temp_files(summaries)
        config_text = make_simple_config_text([[input_path, summary_paths] 
                                               for input_path in input_paths])
        config_path = manager.create_temp_file(config_text)
        if order == "L":
            df = wrapper.compute_rouge(
                config_path, max_ngram=0, lcs=True, length=length, 
                length_unit=length_unit, remove_stopwords=remove_stopwords)
        else:
            order = int(order)
            df = wrapper.compute_rouge(
                config_path, max_ngram=order, lcs=False, length=length, 
                length_unit=length_unit, remove_stopwords=remove_stopwords)
            
        scores = df["rouge-{}".format(order)].values.ravel()[:-1]
        ranked_indices = [i for i in np.argsort(scores)[::-1]]

        candidate_extracts = []
        agg_texts = []
        for i in ranked_indices:
            agg_texts.append(sentences[i])
            candidate_extracts.append("\n".join(agg_texts))
        
        input_paths = manager.create_temp_files(candidate_extracts)
        config_text = make_simple_config_text([[input_path, summary_paths] 
                                               for input_path in input_paths])
        config_path = manager.create_temp_file(config_text)
       
        if order == "L":
            df = wrapper.compute_rouge(
                config_path, max_ngram=0, lcs=True, length=length, 
                length_unit=length_unit, remove_stopwords=remove_stopwords)
        else:
            df = wrapper.compute_rouge(
                config_path, max_ngram=order, lcs=False, length=length, 
                length_unit=length_unit, remove_stopwords=remove_stopwords)
        
        opt_sent_length = np.argmax(
            df["rouge-{}".format(order)].values.ravel()[:-1])
        extract_indices = ranked_indices[:opt_sent_length + 1]
        
        labels = [0] * len(sentences)
        
        for rank, index in enumerate(extract_indices, 1):
            labels[index] = rank


        pairwise_ranks = []
        for i, top_index in enumerate(ranked_indices[:5]):
            for bottom_index in ranked_indices[i+1:]:
                pairwise_ranks.append((int(top_index), int(bottom_index)))

        return labels, pairwise_ranks


def compute_greedy_sequential_extract(sentences, summaries, order, 
                                      length=100, length_unit="word",
                                      remove_stopwords=False):
    
    with TempFileManager() as manager:
        summary_paths = manager.create_temp_files(summaries)
        
        options = [(i, sent) for i, sent in enumerate(sentences)]

        current_indices = []
        current_summary_sents = []
        current_score = 0

        while len(options) > 0:

            candidates = []
            for idx, sent in options:
                candidates.append("\n".join(current_summary_sents + [sent]))
            candidate_paths = manager.create_temp_files(candidates)

            config_text = make_simple_config_text(
                [[cand_path, summary_paths] for cand_path in candidate_paths])
            config_path = manager.create_temp_file(config_text)

            if order == "L":
                df = wrapper.compute_rouge(
                    config_path, max_ngram=0, lcs=True, length=length, 
                    length_unit=length_unit, remove_stopwords=remove_stopwords)
            else:
                order = int(order)
                df = wrapper.compute_rouge(
                    config_path, max_ngram=order, lcs=False, length=length, 
                    length_unit=length_unit, remove_stopwords=remove_stopwords)
                
            scores = df["rouge-{}".format(order)].values.ravel()[:-1]
            ranked_indices = [i for i in np.argsort(scores)[::-1]]
            
            if scores[ranked_indices[0]] > current_score:
                current_score = scores[ranked_indices[0]]
                current_indices.append(options[ranked_indices[0]][0])
                current_summary_sents.append(options[ranked_indices[0]][1])
                options.pop(ranked_indices[0])
            else:
                break

        labels = [0] * len(sentences)
        
        for rank, index in enumerate(current_indices, 1):
            labels[index] = rank
        
        return labels

def compute_greedy_sequential_pairwise_ranks(sentences, summaries, order, 
                                             length=100, length_unit="word",
                                             remove_stopwords=False):
    
    with TempFileManager() as manager:
        summary_paths = manager.create_temp_files(summaries)
        
        options = [(i, sent) for i, sent in enumerate(sentences)]

        current_indices = []
        current_summary_sents = []
        current_score = 0

        from collections import defaultdict
        ranks = defaultdict(list)

        while len(options) > 0:

            candidates = []
            for idx, sent in options:
                candidates.append("\n".join(current_summary_sents + [sent]))
            candidate_paths = manager.create_temp_files(candidates)

            config_text = make_simple_config_text(
                [[cand_path, summary_paths] for cand_path in candidate_paths])
            config_path = manager.create_temp_file(config_text)

            if order == "L":
                df = wrapper.compute_rouge(
                    config_path, max_ngram=0, lcs=True, length=length, 
                    length_unit=length_unit, remove_stopwords=remove_stopwords)
            else:
                order = int(order)
                df = wrapper.compute_rouge(
                    config_path, max_ngram=order, lcs=False, length=length, 
                    length_unit=length_unit, remove_stopwords=remove_stopwords)
                
            scores = df["rouge-{}".format(order)].values.ravel()[:-1]
            ranked_indices = [i for i in np.argsort(scores)[::-1]]
            ranked_options = [options[i][0] for i in ranked_indices]


            if scores[ranked_indices[0]] > current_score:
                current_score = scores[ranked_indices[0]]
                current_indices.append(options[ranked_indices[0]][0])
                current_summary_sents.append(options[ranked_indices[0]][1])
                options.pop(ranked_indices[0])


                print(scores)
                print(ranked_options)


                for i in range(5):
                    for rank in [(ranked_options[i], j) 
                                 for j in ranked_options[i+1:]]:
                        ranks[tuple(sorted(rank))].append(rank)
                for x, y in ranks.items():
                    print(x, y)


                print("")



            else:
                break

        exit()
        labels = [0] * len(sentences)
        
        for rank, index in enumerate(current_indices, 1):
            labels[index] = rank
        
        return labels
