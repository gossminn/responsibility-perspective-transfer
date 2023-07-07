from collections import defaultdict
import json
import os


def make_evaluation_pairs(evaluation_file, pair_file, out_name):
    pairs = read_pairs(pair_file)
    with open(evaluation_file, encoding="utf-8") as f_in, open(out_name, "w", encoding="utf-8") as f_out:
        for line in f_in:
            eval_entry = json.loads(line)
            source = eval_entry["source"]
            targets = pairs[source]
            predictions = eval_entry["targets"]
            f_out.write(json.dumps({"source": source, "targets": targets, "predictions": predictions}) + os.linesep)

def read_pairs(pair_file):
    src_to_tgt_list = defaultdict(list)
    with open(pair_file, encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            src_to_tgt_list[pair["sentence_low"]].append(pair["sentence_high"])
    return src_to_tgt_list


if __name__ == "__main__":
    make_evaluation_pairs("output/questionnaire/questions.group1.sents.jsonl", "output/hum_eval_set_pairs.jsonl", "output/evaluation_pairs_hum_eval_1.jsonl")
    make_evaluation_pairs("output/questionnaire/questions.group2.sents.jsonl", "output/hum_eval_set_2_pairs.jsonl", "output/evaluation_pairs_hum_eval_2.jsonl")
