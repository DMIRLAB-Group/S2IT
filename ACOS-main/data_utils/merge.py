import json
import Levenshtein

from data_utils.subgraph import get_subgraph_single, get_subgraph_span

def read_intermidiate_file_json(data_path):
    def ao_parse(seq):
        pairs = []
        sents = [s.strip() for s in seq.split(' | ')]
        for s in sents:
            try:
                _, a, b = s.split(":")
                a, b = a.strip(), b.strip()
                a = a.replace(', opinion', '')
                b = b.replace(', sentiment', '')
            except ValueError:
                a, b = '', ''
            pairs.append((a, b))
        return pairs

    def oa_parse(seq):
        triplets = []
        sents = [s.strip() for s in seq.split(' | ')]
        for s in sents:
            try:
                _, a, b = s.split(":")
                a, b = a.strip(), b.strip()
                a = a.replace(', aspect', '')
                b = b.replace(', sentiment', '')
            except ValueError:
                a, b = '', ''
            triplets.append((b, a))
        return triplets
    
    preds = []
    result = []
    for i, line in enumerate(open(data_path, 'r', encoding='UTF-8')):
        data = json.loads(line)
        if (i % 2) == 0:
            pred = ao_parse(data["predict"])
        else:
            pred = oa_parse(data["predict"])
        preds.append(pred)
    for i in range(0, len(preds), 2):
        curs = []
        ao_pred_dup, oa_pred_dup = preds[i].copy(), preds[i+1].copy()

        for cur in preds[i]:
            if cur in oa_pred_dup:
                ao_pred_dup.remove(cur)
                oa_pred_dup.remove(cur)
                curs.append(cur)
        if curs == []:
            curs = [('implicit', 'implicit')]
        result.append(curs)
    return result

def find_index(element, sentence):
    def find_string_in_list(target_string, string_list):
        max_distance = 5
        closest_match = None
        closest_distance = float('inf')
        # print(target_string, string_list)

        for index, string in enumerate(string_list):
            distance = Levenshtein.distance(target_string, string)
            if distance < closest_distance:
                closest_distance = distance
                closest_match = index

        if closest_distance <= max_distance:
            return closest_match
        else:
            print(f"字符串 '{target_string}' 在列表中找不到，且没有找到距离在 {max_distance} 以内的匹配。")
            return None

    if len(element) == 1:
        if element[0] == 'implicit':
            index = 10000
        else:
            index = find_string_in_list(element[0], sentence)
        return [index]
    else:
        index_begin = find_string_in_list(element[0], sentence)
        index_end = find_string_in_list(element[-1], sentence)
        return [index_begin, index_end]

def find_subgraph(sentence, dep_tree, index):
    if index[0] == index[1]:
        if index[0] == 10000:
            sub_graph_a = ['nothing']
        else:
            sub_graph_a_1hop = get_subgraph_single(sentence, dep_tree, index[0], 1)
            sub_graph_a_2hop = get_subgraph_single(sentence, dep_tree, index[0], 2)
            sub_graph_a = [sub_graph_a_1hop, sub_graph_a_2hop]
    else:
        sub_graph_a_1hop = get_subgraph_span(sentence, dep_tree, index[0], index[-1], 1)
        sub_graph_a_2hop = get_subgraph_span(sentence, dep_tree, index[0], index[-1], 2)
        sub_graph_a = [sub_graph_a_1hop, sub_graph_a_2hop]
    return sub_graph_a
