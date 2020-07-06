import os
import pkg_resources
from subprocess import check_output
import re
import pandas as pd


AVG_RECALL_PATT = r"X ROUGE-{} Average_R: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"
AVG_PRECISION_PATT = r"X ROUGE-{} Average_P: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"
AVG_FSCORE_PATT = r"X ROUGE-{} Average_F: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"

def compute_rouge(config_path, show_all=True, max_ngram=4, lcs=False, 
                  stemmer=True, set_length = False,length=100, length_unit="word",
                  number_of_samples=1000, scoring_formula="A",
                  remove_stopwords=False, return_conf=False):
    rouge_path = pkg_resources.resource_filename(
        'rouge_papier_v2', os.path.join('data', 'ROUGE-1.5.5.pl'))
    rouge_data_path = pkg_resources.resource_filename(
        'rouge_papier_v2', os.path.join('rouge_data'))

    #-n 4 -m -a -l 100 -x -c 95
    #-r 1000 -f A -p 0.5 -t 0
    args = ["perl", rouge_path, "-e", rouge_data_path, "-a"]

    if max_ngram > 0:
        args.extend(["-n", str(max_ngram)])

    if not lcs:
        args.append("-x")

    if show_all:
        args.append("-d")
           
    if stemmer:
        args.append("-m")

    if remove_stopwords:
        args.append("-s")

    if set_length:
        if length_unit == "word":
            args.extend(["-l", str(length)])
        elif length_unit == "byte":
            args.extend(["-b", str(length)])
        else:
            raise Exception(
                "length_unit must be either 'word' or 'byte' but found {}".format(
                    length_unit))

    args.extend(["-r", str(number_of_samples)])

    if scoring_formula not in ["A", "B"]:
        raise Exception(
            "scoring_formula must be either 'A' or 'B' but found {}".format(
                scoring_formula))
    else:
        args.extend(["-f", scoring_formula])

    args.extend(["-z", "SPL", config_path])

    output = check_output(" ".join(args), shell=True).decode("utf8")
    dataframes = []
    confs = []
    avgfs = []
    for r in range(1, max_ngram + 1):
        o, avgf,conf = convert_output2(output, r)
        dataframes.append(o)
        confs.append(conf)
        avgfs.append(avgf)
    if lcs:
        o, avgf,conf = convert_output2(output, "L")
        dataframes.append(o)
        confs.append(conf)
        avgfs.append(avgf)

    df = pd.concat(dataframes, axis=1)
    if return_conf:
        conf = pd.concat(confs, axis=0)
        return df, avgfs,conf
    else:
        return df,avgfs

def convert_output(output, rouge=1):
    data = []
    avg_recall_patt = AVG_RECALL_PATT.format(rouge)
    patt = r"X ROUGE-{} Eval (.*?) R:(.*?) P:(.*?) F:(.*?)$".format(rouge)
    for match in re.findall(patt, output, flags=re.MULTILINE):
        name, recall, prec, fmeas = match
        data.append((name, float(recall)))
    match = re.search(avg_recall_patt, output, flags=re.MULTILINE)
    avg_recall = float(match.groups()[0])
    lower_conf = float(match.groups()[1])
    upper_conf = float(match.groups()[2])
    data.append(("average", avg_recall))

    df = pd.DataFrame(data, columns=["name", "rouge-{}".format(rouge)])
    df.set_index("name", inplace=True)
    conf = pd.DataFrame([[lower_conf, upper_conf]], 
                        columns=["95% conf. lb.", "95% conf. ub."])
    conf.index = ["rouge-{}".format(rouge)]
    return df, conf


def convert_output2(output, rouge=1):
    data = []
    avg_recall_patt = AVG_RECALL_PATT.format(rouge)
    avg_precision_patt = AVG_PRECISION_PATT.format(rouge)
    avg_fscore_patt = AVG_FSCORE_PATT.format(rouge)

    patt = r"X ROUGE-{} Eval (.*?) R:(.*?) P:(.*?) F:(.*?)$".format(rouge)
    for match in re.findall(patt, output, flags=re.MULTILINE):
        name, recall, prec, fmeas = match
        data.append((name, float(recall),float(prec),float(fmeas)))

    match = re.search(avg_recall_patt, output, flags=re.MULTILINE)
    avg_recall = float(match.groups()[0])

    match = re.search(avg_precision_patt, output, flags=re.MULTILINE)
    avg_precision = float(match.groups()[0])

    match = re.search(avg_fscore_patt, output, flags=re.MULTILINE)
    avg_fscore = float(match.groups()[0])


    data.append(("average", avg_recall,avg_precision,avg_fscore))

    df = pd.DataFrame(data, columns=["name", "rouge-{}-r".format(rouge),"rouge-{}-p".format(rouge),"rouge-{}-f".format(rouge)])
    df.set_index("name", inplace=True)
    # conf = pd.DataFrame([[lower_conf, upper_conf]], 
    #                     columns=["95% conf. lb.", "95% conf. ub."])
    # conf.index = ["rouge-{}".format(rouge)]
    conf = ''
    return df,avg_fscore,conf
