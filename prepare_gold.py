import glob
import json
from collections import defaultdict

import pandas as pd


def prepare_gold():

    sentences_to_meta = {}
    for qmeta in glob.glob("data/questionnaire-stimuli/*.meta.jsonl"):
        qmeta_df = pd.read_json(qmeta, orient="records", lines=True)
        for _, row in qmeta_df.iterrows():
            sentences_to_meta[row["sentence_data"]["sentence_str"]] = row["sentence_data"]
            del row["sentence_data"]["sentence_str"]

    print(len(sentences_to_meta))

    perception_df = (
        pd.read_json("data/questionnaire-combined.jsonl", orient="records", lines=True)
        .sample(frac=1., random_state=1996)
    )

    data_out_full = []
    data_out_simple = []
    grouped_by_event = defaultdict(list)
    for i, (_, row) in enumerate(perception_df.iterrows()):
        if i < 300:
            split = "train+dev"
        else:
            split = "test"

        data_out_full.append({
            "split": split,
            "sentence": row["sentence"],
            "perception": {k: v for k, v in row.items() if k in perception_df.columns[3:]},
            "meta": sentences_to_meta[row["sentence"]]
        })
        data_out_simple.append({
            "split": split,
            "sentence": row["sentence"],
            "blame_perception": "low" if row["B:suspect"] < 0 else "high"
        })
        grouped_by_event[sentences_to_meta[row["sentence"]]["event_id"]].append({
            "split": split,
            "sentence": row["sentence"],
            "document_id": sentences_to_meta[row["sentence"]]["text_id"],
            "blame_perception": "low" if row["B:suspect"] < 0 else "high"
        })

    pd.DataFrame(data_out_full).to_json("output/preproc/gold.full.jsonl", orient="records", lines=True)
    pd.DataFrame(data_out_simple).to_json("output/preproc/gold.simple.jsonl", orient="records", lines=True)
    with open("output/preproc/grouped_by_event.json", "w", encoding="utf-8") as f:
        json.dump(grouped_by_event, f, indent=4)   

if __name__ == "__main__":
    prepare_gold()
