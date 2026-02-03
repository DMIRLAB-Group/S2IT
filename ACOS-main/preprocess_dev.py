import os
from data_utils.parse import *
from data_utils.subgraph import *

# Prepare step1 dev dataset
for x in ['laptop', 'restaurant']:
    for y in ['dev', 'test']:
        data_path = f'ACOS-main/acos/{x}/{y}.json'
        save_data_path = f'data/acos/extraction/{x}/{y}.json'
        if not os.path.exists(f'data/acos/extraction/{x}/'):
            os.makedirs(f'data/acos/extraction/{x}/')

        all_inputs, clean_sents, all_targets, relations = read_file_json(data_path)
        datas = []

        for i in range(len(all_inputs)):
            input = all_inputs[i]
            clean_sent = clean_sents[i]
            sentence = "sentence: " + " ".join(input)
            target_seq = all_targets[i]
            relation = relations[i]

            # Construct dependency relation sequence
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

        with open(save_data_path, "w") as json_file:
            json.dump(datas, json_file, indent=4, ensure_ascii=False)

# Prepare step2 dev dataset
for x in ['laptop', 'restaurant']:
    for y in ['dev', 'test']:
        data_path = f'ACOS-main/acos/{x}/{y}.json'
        save_data_path = f'data/acos/classification/{x}/{y}.json'
        if not os.path.exists(f'data/acos/classification/{x}/'):
            os.makedirs(f'data/acos/classification/{x}/')

        all_inputs, clean_sents, all_targets, relations = read_file_json(data_path)
        datas = []

        for i in range(len(all_inputs)):
            input = all_inputs[i]
            clean_sent = clean_sents[i]
            sentence = "sentence: " + " ".join(input)
            target_seq = all_targets[i]
            relation = relations[i]
            target = read_target(input, target_seq, 1)

            # Construct dependency relation sequence
            dep_tree = []
            for rel in relation:
                dep_tree.append(rel["head"]-1)
            subgraph = get_subgraph(input, clean_sent, target_seq, dep_tree)

            output = ["aspect: " + a + ", opinion: " + b + ", category: " + c + ", sentiment: " + d for (a, b, c, d) in target]
            output = " | ".join(output)

            # (aspect, opinion, category, sentiment)
            data = {}
            insturct = "task: (classification (aspect, opinion) to (category, sentiment)): Given a sentence, related dependency relations (will be presented in the form of subgraph) and (aspect, opinion) candidates, determine the category of the aspect and the sentiment (positive, neutral, negative) of the opinion and return the quadruple(aspect, opinion, category, sentiment). "

            subgraph_seq = [f"aspect: {a}, which is connected to ({subgraph['all_subgraph_a'][i][0]}) within one hop. opinion: {b}, which is connected to ({subgraph['all_subgraph_o'][i][0]}) within one hop." for i, (a, b, _, _) in enumerate(target)]
            subgraph_seq = " \n\nsubgraph: " + " | ".join(subgraph_seq)
            
            candidate = [f"aspect: {a}, opinion: {b}" for i, (a, b, _, _) in enumerate(target)]
            candidate = " | ".join(candidate)
            candidate = " \n\ncandidate: " + candidate

            data["instruction"] = insturct
            data["input"] = sentence + subgraph_seq + candidate
            data["output"] = output
            datas.append(data)

        with open(save_data_path, "w") as json_file:
            json.dump(datas, json_file, indent=4, ensure_ascii=False)