import os
import json

from data_utils.parse import *
from data_utils.subgraph import *
from data_utils.merge import *

dataset = os.environ.get('dataset')

intermidiate_result_path = f'./ao_result/generated_predictions.jsonl'
save_result_path = f'./ao_result/save_result.json'
data_path = f'ACOS-main/acos/{dataset}/test.json'

if __name__ == "__main__":
    intermidiate_result = read_intermidiate_file_json(intermidiate_result_path)
    all_inputs, clean_sents, all_targets, relations = read_file_json(data_path)
    assert len(intermidiate_result) == len(all_inputs)
    datas = []

    all_subgraph_a, all_subgraph_o = [], []
    for i in range(len(all_inputs)):
        data = {}
        input = all_inputs[i]
        clean_sent = clean_sents[i]
        sentence = "sentence: " + " ".join(input)
        target_seq = all_targets[i]
        relation = relations[i]
        target = read_target(input, target_seq, 1)

        dep_tree = []
        for rel in relation:
            dep_tree.append(rel["head"]-1)

        inter_result = intermidiate_result[i]
        # for j in inter_result:
        #     aspect = j[0].split(" ")
        #     opinion = j[1].split(" ")
        #     aspect_index = find_index(aspect, input)
        #     opinion_index = find_index(opinion, input)
        #     print(aspect_index)

        #     sub_graph_a = find_subgraph(clean_sent, dep_tree, aspect_index)
        #     sub_graph_o = find_subgraph(clean_sent, dep_tree, opinion_index)
        #     all_subgraph_a.append(sub_graph_a)
        #     all_subgraph_o.append(sub_graph_o)
        inter_targets = []
        for j in inter_result:
            aspect = j[0].split(" ")
            opinion = j[1].split(" ")
            aspect_index = find_index(aspect, input)
            opinion_index = find_index(opinion, input)
            inter_target = [aspect_index, "", "", opinion_index]
            inter_targets.append(inter_target)

        subgraph = get_subgraph(input, clean_sent, inter_targets, dep_tree)


        insturct = "task: (classification (aspect, opinion) to (category, sentiment)): Given a sentence, related dependency relations (will be presented in the form of subgraph) and (aspect, opinion) candidates, determine the category of the aspect and the sentiment (positive, neutral, negative) of the opinion and return the quadruple(aspect, opinion, category, sentiment). "

        subgraph_seq = [f"aspect: {a}, which is connected to ({subgraph['all_subgraph_a'][i][0]}) within one hop. opinion: {b}, which is connected to ({subgraph['all_subgraph_o'][i][0]}) within one hop." for i, (a, b) in enumerate(inter_result)]
        subgraph_seq = " \n\nsubgraph: " + " | ".join(subgraph_seq)

        candidate = [f"aspect: {a}, opinion: {b}" for i, (a, b) in enumerate(inter_result)]
        candidate = " | ".join(candidate)
        candidate = " \n\ncandidate: " + candidate

        output = ["aspect: " + a + ", opinion: " + b + ", category: " + c + ", sentiment: " + d for (a, b, c, d) in target]
        output = " | ".join(output)

        data["instruction"] = insturct
        data["input"] = sentence + subgraph_seq + candidate
        data["output"] = output
        datas.append(data)

    with open(save_result_path, "w") as json_file:
        json.dump(datas, json_file, indent=4, ensure_ascii=False)