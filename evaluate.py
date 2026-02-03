import os
import json

dataset = os.environ.get('dataset')

def read_file_json(data_path):
    def quad_parse(seq):
        pairs = []
        sents = [s.strip() for s in seq.split(' | ')]
        for s in sents:
            try:
                _, a, b, c, d = s.split(":")
                a, b, c, d = a.strip(), b.strip(), c.strip(), d.strip()
                a = a.replace(', opinion', '')
                b = b.replace(', category', '')
                c = c.replace(', sentiment', '')
                d = d
            except ValueError:
                a, b, c, d = '', '', '', ''
            pairs.append((a, b, c, d))
        return pairs
    labels = []
    preds = []
    for i, line in enumerate(open(data_path, 'r', encoding='UTF-8')):
        data = json.loads(line)
        label = quad_parse(data["label"])
        labels.append(label)
        pred = quad_parse(data["predict"])
        preds.append(pred)
    return labels, preds


if __name__ == "__main__":
    result_path = f"{dataset}_result/generated_predictions.jsonl"
    labels, preds = read_file_json(result_path)
    assert len(labels) == len(preds)
    labels_num, preds_num, correct_num = 0, 0, 0
    for label, pred in zip(labels, preds):
        labels_num += len(label)
        preds_num += len(pred)
        for p in pred:
            if p in label:
                correct_num += 1
                label.remove(p)
    precision = correct_num / preds_num
    recall = correct_num / labels_num
    f1 = 2 * precision * recall / (precision + recall)
    print(f"precision: {precision}, recall: {recall}, f1: {f1}")
    