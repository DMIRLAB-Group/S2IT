import json
from tqdm import tqdm
import os
import csv
import string
import re
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
# stanza.download('en')
import stanza
nlp = stanza.Pipeline('en',
                      dir="stanza_resources/stanza-en/",
                      download_method=None,
                      processors = 'tokenize,pos,lemma,depparse,ner,constituency',
                      tokenize_pretokenized=True,
                      ) # initialize English neural pipeline

def find_substring_indices(substring, string):
    indices = []
    len_substring = len(substring)
    len_string = len(string)
    for i in range(len_string - len_substring + 1):
        if string[i:i+len_substring] == substring:
            for j in range(i, i+len_substring):
                indices.append(j)
    return indices


for x in ['laptop', 'restaurant']:
    for y in ['train', 'dev', 'test']:
        data_path = f'ACOS-main/data/{x}-ACOS/{x}_quad_{y}.tsv'
        json_file_path = f'ACOS-main/acos/{x}/{y}.json'
        if not os.path.exists(f'ACOS-main/acos/{x}/'):
            os.makedirs(f'ACOS-main/acos/{x}/')
        count = 0
        with open(data_path, 'r', encoding='UTF-8') as fp:
            for line in fp:
                count = count + 1
        pbar = tqdm(total=count)
        with open(data_path, 'r', encoding='UTF-8') as fp:
            reader = csv.reader(fp, delimiter='\t')
            datas = []
            for line in reader:
                tuples = []
                data = {}
                if line != '':
                    sentence = line[0]
                    raw_labels = line[1:]
                    for label in raw_labels:
                        label = label.split(" ")
                        label[0] = eval('[' + label[0] + ']')
                        if label[0][0] == label[0][1]:
                            label[0][1] = label[0][1] + 1
                        label[0] = list(range(label[0][0], label[0][1])) if label[0][0] != -1 else [10000]

                        label[3] = eval('[' + label[3] + ']')
                        if label[3][0] == label[3][1]:
                            label[3][1] = label[3][1] + 1
                        label[3] = list(range(label[3][0], label[3][1])) if label[3][0] != -1 else [10000]

                        label[2] = eval(label[2])
                        label = list(label)
                        tuples.append(label)
                    sentence = sentence.split()


                    # 定义正则表达式：
                    # 1. [^\x00-\x7F] 匹配所有非 ASCII 字符
                    # 2. [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~] 匹配所有 ASCII 标点符号
                    pattern = r'[^\w\s]|[^\x00-\x7F]'

                    # 对列表中的每个字符串进行标点符号和非 ASCII 字符的删除，并过滤掉空字符串
                    cleaned_data = [re.sub(pattern, '', s).strip() for s in sentence if re.sub(pattern, '', s).strip()]

                    doc = nlp(" ".join(cleaned_data))
                    ws = []


                    sents = []
                    for sent in doc.sentences:
                        # print(sent.constituency)
                        for word in sent.words:
                            w = {}
                            w['id'] = word.id
                            w['text'] = word.text
                            sents.append(w['text'])
                            w['upos'] = word.upos
                            w['xpos'] = word.xpos
                            w['head'] = word.head
                            w['deprel'] = word.deprel
                            ws.append(w)
                    
                    data['sentence'] = sentence
                    data['cleaned_data'] = cleaned_data
                    data['label'] = tuples
                    data['relations'] = ws
                    data['constituency'] = str(sent.constituency)

                    datas.append(data)
                pbar.update(1)
                # exit()

        with open(json_file_path, "w") as json_file:
                json.dump(datas, json_file, indent=4, ensure_ascii=False)
        pbar.close()
print("finish!")

