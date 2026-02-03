import os

from data_utils.parse import *
from data_utils.subgraph import *

for x in ['laptop', 'restaurant']:
    for y in ['train']:
        data_path = f'ACOS-main/acos/{x}/{y}.json'
        save_data_path = f'data/acos/training/{x}/{y}.json'
        if not os.path.exists(f'data/acos/training/{x}/'):
            os.makedirs(f'data/acos/training/{x}/')

        all_inputs, clean_sents, all_targets, relations = read_file_json(data_path)
        datas = []

        for i in range(len(all_inputs)):
            input = all_inputs[i]
            clean_sent = clean_sents[i]
            sentence = "sentence: " + " ".join(input)
            target_seq = all_targets[i]
            relation = relations[i]

            # remove invalid example
            skip = False
            for (a, b, _, _) in read_target(input, target_seq, 1):
                if a == "_" or a == "." or b == "_" or b == ".":
                    skip = True

            if skip:
                continue

            #TODO: Construct dependency relation sequence
            dep_sequence = []
            for rel in relation:
                # triplet
                now_word = rel["text"]
                head_word = clean_sent[rel["head"]-1] if rel["head"]!=0 else "root"
                dep_rel = rel["deprel"]

                edge = f"{head_word} depend {now_word}" if dep_rel not in mod else f"{head_word} modify {now_word}"
                if dep_rel not in pruning:
                    dep_sequence.append(edge)
            dep_sequence = " \n\ndependency relation: " + " | ".join(dep_sequence)

            #TODO: Construct dependency relation sequence
            dep_tree = []
            for rel in relation:
                dep_tree.append(rel["head"]-1)

            subgraph = get_subgraph(input, clean_sent, target_seq, dep_tree)
            # print(subgraph)

            # (aspect, opinion)
            data = {}
            insturct = "task: (extract aspect, opinion): Given a sentence and related dependency relations, extract aspect and opinion (both implicit and explicit) from the sentence and return pair(aspect, opinion). Pay attention to the one or multi hop dependency relationships between aspect and opinion. "

            output = ["aspect: " + a + ", opinion: " + b for (a, b, _, _) in read_target(input, target_seq, 1)]
            output = " | ".join(output)

            data["instruction"] = insturct
            data["input"] = sentence + dep_sequence
            data["output"] = output
            datas.append(data)

            # (opinion, aspect)
            data = {}
            insturct = "task: (extract opinion, aspect): Given a sentence and related dependency relations, extract opinion and aspect (both implicit and explicit) from the sentence and return pair(opinion, aspect). Pay attention to the one or multi hop dependency relationships between aspect and opinion. "

            output = ["opinion: " + b + ", aspect: " + a for (a, b, _, _) in read_target(input, target_seq, 0)]
            output = " | ".join(output)

            data["instruction"] = insturct
            data["input"] = sentence + dep_sequence
            data["output"] = output
            datas.append(data)

            # (link aspect to opinion)
            data = {}
            insturct = "task: (linking aspect to opinion): Given a sentence, related dependency relations and known aspects, determine the opinion (both implicit and explicit) related to the each aspect from dependency relation and return the pair(aspect, opinion). "

            candidate = ["aspect: " + a for (a, _, _, _) in read_target(input, target_seq, 1)]
            candidate = " \n\ncandidates: " + " | ".join(candidate)

            output = ["aspect: " + a + ", opinion: " + b for (a, b, _, _) in read_target(input, target_seq, 1)]
            output = " | ".join(output)

            data["instruction"] = insturct
            data["input"] = sentence + dep_sequence + candidate
            data["output"] = output
            datas.append(data)

            # (link opinion to aspect)
            data = {}
            insturct = "task: (linking opinion to aspect): Given a sentence, related dependency relations and known opinions, determine the aspect (both implicit and explicit) related to the each opinion from dependency relation and return the pair(opinion, aspect). "

            candidate = ["opinion: " + b for (_, b, _, _) in read_target(input, target_seq, 0)]
            candidate = " \n\ncandidates: " + " | ".join(candidate)

            output = ["opinion: " + b + ", aspect: " + a for (a, b, _, _) in read_target(input, target_seq, 0)]
            output = " | ".join(output)

            data["instruction"] = insturct
            data["input"] = sentence + dep_sequence + candidate
            data["output"] = output
            datas.append(data)

            # (aspect, opinion, category, sentiment)
            data = {}
            insturct = "task: (classification (aspect, opinion) to (category, sentiment)): Given a sentence, related dependency relations (will be presented in the form of subgraph) and (aspect, opinion) candidates, determine the category of the aspect and the sentiment (positive, neutral, negative) of the opinion and return the quadruple(aspect, opinion, category, sentiment). "

            subgraph_seq = [f"aspect: {a}, which is connected to ({subgraph['all_subgraph_a'][i][0]}) within one hop. opinion: {b}, which is connected to ({subgraph['all_subgraph_o'][i][0]}) within one hop." for i, (a, b, _, _) in enumerate(read_target(input, target_seq, 1))]
            subgraph_seq = " \n\nsubgraph: " + " | ".join(subgraph_seq)
            
            candidate = [f"aspect: {a}, opinion: {b}" for (a, b, _, _) in read_target(input, target_seq, 1)]
            candidate = " | ".join(candidate)
            candidate = " \n\ncandidate: " + candidate

            output = ["aspect: " + a + ", opinion: " + b + ", category: " + c + ", sentiment: " + d for (a, b, c, d) in read_target(input, target_seq, 1)]
            output = " | ".join(output)

            data["instruction"] = insturct
            data["input"] = sentence + subgraph_seq + candidate
            data["output"] = output
            datas.append(data)

            # classification aspect to category
            data = {}
            insturct = "task: (classification aspect to category): Given a sentence, related dependency relations (will be presented in the form of subgraph) and known aspects (both implicit and explicit) , determine the category related to the each aspects from dependency relation and return pair (aspect, category). "

            subgraph_seq = [f"aspect: {a}, which is connected to ({subgraph['all_subgraph_a'][i][0]}) within one hop." for i, (a, b, _, _) in enumerate(read_target(input, target_seq, 1))]
            subgraph_seq = " \n\nsubgraph: " + " | ".join(subgraph_seq)
            
            candidate = ["aspect: " + a for (a, _, _, _) in read_target(input, target_seq, 1)]
            candidate = " \n\ncandidates: " + " | ".join(candidate)

            output = ["aspect: " + a + ", category: " + c for (a, b, c, d) in read_target(input, target_seq, 1)]
            output = " | ".join(output)

            data["instruction"] = insturct
            data["input"] = sentence + subgraph_seq + candidate
            data["output"] = output
            datas.append(data)

            # classification aspect to sentiment
            data = {}
            insturct = "task: (classification aspect to sentiment): Given a sentence, related dependency relations (will be presented in the form of subgraph) and known aspects (both implicit and explicit) , determine the sentiment related to the each aspects from dependency relation and return pair (aspect, sentiment). "

            subgraph_seq = [f"aspect: {a}, which is connected to ({subgraph['all_subgraph_a'][i][0]}) within one hop." for i, (a, b, _, _) in enumerate(read_target(input, target_seq, 1))]
            subgraph_seq = " \n\nsubgraph: " + " | ".join(subgraph_seq)
            
            candidate = ["aspect: " + a for (a, _, _, _) in read_target(input, target_seq, 1)]
            candidate = " \n\ncandidates: " + " | ".join(candidate)

            output = ["aspect: " + a + ", sentiment: " + d for (a, b, c, d) in read_target(input, target_seq, 1)]
            output = " | ".join(output)

            data["instruction"] = insturct
            data["input"] = sentence + subgraph_seq + candidate
            data["output"] = output
            datas.append(data)

            # classification opinion to category
            data = {}
            insturct = "task: (classification opinion to category): Given a sentence, related dependency relations (will be presented in the form of subgraph) and known opinions (both implicit and explicit) , determine the category related to the each opinions from dependency relation and return pair (opinion, category). "

            subgraph_seq = [f"opinion: {b}, which is connected to ({subgraph['all_subgraph_o'][i][0]}) within one hop." for i, (a, b, _, _) in enumerate(read_target(input, target_seq, 0))]
            subgraph_seq = " \n\nsubgraph: " + " | ".join(subgraph_seq)
            
            candidate = ["opinion: " + b for (_, b, _, _) in read_target(input, target_seq, 0)]
            candidate = " \n\ncandidates: " + " | ".join(candidate)

            output = ["opinion: " + b + ", category: " + c for (a, b, c, d) in read_target(input, target_seq, 1)]
            output = " | ".join(output)

            data["instruction"] = insturct
            data["input"] = sentence + subgraph_seq + candidate
            data["output"] = output
            datas.append(data)

            # classification opinion to sentiment
            data = {}
            insturct = "task: (classification opinion to sentiment): Given a sentence, related dependency relations (will be presented in the form of subgraph) and known opinions (both implicit and explicit) , determine the sentiment related to the each opinions from dependency relation and return pair (opinion, sentiment). "

            subgraph_seq = [f"opinion: {b}, which is connected to ({subgraph['all_subgraph_o'][i][0]}) within one hop." for i, (a, b, _, _) in enumerate(read_target(input, target_seq, 0))]
            subgraph_seq = " \n\nsubgraph: " + " | ".join(subgraph_seq)
            
            candidate = ["opinion: " + b for (_, b, _, _) in read_target(input, target_seq, 0)]
            candidate = " \n\ncandidates: " + " | ".join(candidate)

            output = ["opinion: " + b + ", sentiment: " + d for (a, b, c, d) in read_target(input, target_seq, 1)]
            output = " | ".join(output)

            data["instruction"] = insturct
            data["input"] = sentence + subgraph_seq + candidate
            data["output"] = output
            datas.append(data)
        with open(save_data_path, "w") as json_file:
            json.dump(datas, json_file, indent=4, ensure_ascii=False)