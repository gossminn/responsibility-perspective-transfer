import json
import random
from collections import defaultdict

from tqdm import tqdm

import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


random.seed(20010727)

DATA_TEST_RATIO = 0.1
DATA_DEV_RATIO = 0.05


def collect_sentences():
    with open("data/all_relevant_frames.json", encoding="utf-8") as f:
        sentences = json.load(f)
    
    sents_to_records = defaultdict(dict)
    all_records = []
    for cx_type, cx_instances in sentences.items():
        for cxi in cx_instances:
            sentence = cxi["sentence_str"]
            del cxi["sentence_str"]
            record = {"split": None, "sentence": sentence, "meta": cxi}
            sents_to_records[sentence][cx_type] = record
            all_records.append(record)

    print("#unique sentences:", len(sents_to_records))
    print("#total records:", len(all_records))
    return sents_to_records, all_records

def prepare_silver():
    sents_to_records, _ = collect_sentences()

    print("Loading models")
    
    model_names = {
        "B:suspect": "predict-perception-bertino-blame-assassin",
        "B:victim": "predict-perception-bertino-blame-victim",
        "B:concept": "predict-perception-bertino-blame-concept",
        "B:object": "predict-perception-bertino-blame-object",
        "B:none": "predict-perception-bertino-blame-none",
        "C:human": "predict-perception-bertino-cause-human",
        "C:object": "predict-perception-bertino-cause-object",
        "C:none": "predict-perception-bertino-cause-none",
        "C:concept": "predict-perception-bertino-cause-concept",
        "F:murderer": "predict-perception-bertino-focus-assassin",
        "F:victim": "predict-perception-bertino-focus-victim",
        "F:object": "predict-perception-bertino-focus-object",
        "F:concept": "predict-perception-bertino-focus-concept",
    }
    
    pipes = {}
    for key, model_name in model_names.items():
        tokenizer = AutoTokenizer.from_pretrained(f"gossminn/{model_name}", use_auth_token=True)
        model = AutoModelForSequenceClassification.from_pretrained(f"gossminn/{model_name}", use_auth_token=True)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, function_to_apply="none")
        pipes[key] = pipe

    data_out_full = []
    data_out_simple = []
    grouped_by_event = defaultdict(list)
    for _, records in tqdm(sents_to_records.items()):
        split_rand = random.random()
        if split_rand < DATA_TEST_RATIO:
            split = "test"
        elif split_rand < DATA_TEST_RATIO + DATA_DEV_RATIO:
            split = "dev"
        else:
            split = "train"
        record = random.choice(list(records.values()))
        sentence = record["sentence"]
        predictions = {k: pipe(sentence)[0]["score"] for k, pipe in pipes.items()}
        data_out_full.append({
            "split": split,
            "sentence": record["sentence"],
            "perception": predictions,
            "meta": record["meta"]
        })
        data_out_simple.append({
            "split": split,
            "sentence": record["sentence"],
            "blame_perception": "low" if predictions["B:suspect"] < 0 else "high"
        })
        grouped_by_event[record["meta"]["event_id"]].append({
            "split": split,
            "sentence": record["sentence"],
            "document_id": record["meta"]["text_id"],
            "blame_perception": "low" if predictions["B:suspect"] < 0 else "high"
        })
    
    pd.DataFrame(data_out_full).to_json("output/preproc/silver.full.jsonl", orient="records", lines=True)
    pd.DataFrame(data_out_simple).to_json("output/preproc/silver.simple.jsonl", orient="records", lines=True)
    with open("output/preproc/silver.grouped_by_event.json", "w", encoding="utf-8") as f:
        json.dump(grouped_by_event, f, indent=4)   

def filter_silver():

    gold_sentences = pd.read_json("output/preproc/gold.simple.jsonl", orient="records", lines=True)["sentence"].to_list()
    
    df_full = pd.read_json("output/preproc/silver.full.jsonl", orient="records", lines=True)
    df_simple = pd.read_json("output/preproc/silver.simple.jsonl", orient="records", lines=True)
    with open("output/preproc/silver.grouped_by_event.json", encoding="utf-8") as f:
        grouped_by_event = json.load(f)

    df_full_ = df_full[~df_full["sentence"].isin(gold_sentences)]
    df_simple_ = df_simple[~df_simple["sentence"].isin(gold_sentences)]
    grouped_by_event_ = {ev: [r for r in recs if r["sentence"] not in gold_sentences] for ev, recs in grouped_by_event.items()}

    df_full_.to_json("output/preproc/silver.full.filtered.jsonl", orient="records", lines=True)
    df_simple_.to_json("output/preproc/silver.simple.filtered.jsonl", orient="records", lines=True)

    with open("output/preproc/silver.grouped_by_event.filtered.json", "w", encoding="utf-8") as f:
        json.dump(grouped_by_event_, f, indent=4)   


if __name__ == "__main__":
    # prepare_silver()
    filter_silver()
