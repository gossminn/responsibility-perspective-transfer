import json
import os
import pandas as pd


def add_meta_info():
    meta_df = pd.concat((
        pd.read_csv("data/split_main.events.csv", dtype={"event:id": int}),
        pd.read_csv("data/split_dev10.events.csv", dtype={"event:id": int})
    ))

    input_files = [
        ("gold.full", "gold.simple"), 
        ("silver.full", "silver.simple"), 
        ("silver.full.filtered", "silver.simple.filtered")
    ]

    for f_full, f_simple in input_files:
        with open(f"output/preproc/{f_full}.jsonl", encoding="utf-8") as ff, open(f"output/preproc/{f_simple}.jsonl", encoding="utf-8") as fs:
            with open(f"output/preproc/{f_full}.w_meta.jsonl", "w", encoding="utf-8") as ffo, open(f"output/preproc/{f_simple}.w_meta.jsonl", "w", encoding="utf-8") as fso:
                for lf, ls in zip(ff, fs):
                    lf_data = json.loads(lf)
                    ls_data = json.loads(ls)
                    event_id = lf_data["meta"]["event_id"]
                    meta_row = meta_df[meta_df["event:id"] == event_id].iloc[0]
                    victim_name = meta_row["victim:name"]
                    killer_name = meta_row["attacker:name"]
                    killer_victim_rel = meta_row["attacker:relation_to_victim"]
                    weapon = meta_row["weapon"]
                    municipality = meta_row["event:location"]
                    sem_location = meta_row["event:semantic_location"]

                    sentence_w_meta = f"frase: {lf_data['sentence']} --- evento: nome della vittima = {victim_name}, nome dell'aggressore = {killer_name}, relazione = {killer_victim_rel}, arma = {weapon}, commune = {municipality}, luogo = {sem_location}"
                    lf_data["sentence_w_meta"] = ls_data["sentence_w_meta"] = sentence_w_meta
                    lf_data["event_information"] = meta_row.to_dict()
                    # todo: fix this problem in the orig RAI processing script
                    lf_data["event_information"]["attacker:occupation"] = meta_row["Unnamed: 20"]
                    del lf_data["event_information"]["Unnamed: 20"]
                    del lf_data["event_information"]["Unnamed: 0"]

                    ffo.write(json.dumps(lf_data) + os.linesep)
                    fso.write(json.dumps(ls_data) + os.linesep)
                

if __name__ == "__main__":
    add_meta_info()