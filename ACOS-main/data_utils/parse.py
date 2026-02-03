import json
import re

polarity2word = {2: "positive", 0: "negative", 1: "neutral"}
mod = ["amod", "nsubj", "advmod", "nmod", "obl:tmod", "nmod:poss", "obl:npmod", "nmod:npmod", "nummod", "nmod:tmod", "compound", "acl", "csubj"]
# pruning = ["punct"]
pruning = []

def read_file_json(data_path):
    """·
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, clean_sents, labels, relations, constituency = [], [], [], [], []

    with open(data_path, 'r', encoding='UTF-8') as fp:
        datas = json.load(fp)
        for i, data in enumerate(datas):
            sent = data['sentence']
            sents.append(sent)
            clean_sents.append(data['cleaned_data'])
            label = data['label']
            label = [tuple(sublist) for sublist in label]
            label.sort(key=lambda x: (x[0][-1], x[1][-1]))
            labels.append(label)
            relations.append(data['relations'])
            # constituency.append(data['constituency'])

    return sents, clean_sents, labels, relations

def read_target(input, target_seq, aspect_first):
    if aspect_first:
        target_seq.sort(key=lambda x: (x[0][-1], x[3][-1]))
    else:
        target_seq.sort(key=lambda x: (x[3][-1], x[1][-1]))
    
    quad = []
    for tri in target_seq:
        if tri[0][0] == 10000:
            a = 'implicit'
        elif len(tri[0]) == 1:
            a = input[tri[0][0]]
        else:
            st, ed = tri[0][0], tri[0][-1]
            a = ' '.join(input[st: ed + 1])
        if tri[3][0] == 10000:
            b = 'implicit'
        elif len(tri[3]) == 1:
            b = input[tri[3][0]]
        else:
            st, ed = tri[3][0], tri[3][-1]
            b = ' '.join(input[st: ed + 1])
        # c = " ".join(tri[1].split("#")).lower()
        c = " ".join(re.split(r'[#_]', tri[1])).lower()
        d = polarity2word[tri[2]]
        quad.append((a, b, c, d))
    
    return quad