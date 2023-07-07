import os
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def autoeval():
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
    
    for dimension, model_name in model_names.items():
        tqdm.write(dimension)

        tokenizer = AutoTokenizer.from_pretrained(f"gossminn/{model_name}", use_auth_token=True)
        model = AutoModelForSequenceClassification.from_pretrained(f"gossminn/{model_name}", use_auth_token=True)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, function_to_apply="none")

        score_table = []
        with open("output/evaluation_pairs_hum_eval_1.jsonl", encoding="utf-8") as f:
            for line in tqdm(f.readlines()):
                item_scores = {}
                item_sentences = json.loads(line)
                
                # source
                source_score = pipe(item_sentences["source"])[0]["score"]
                item_scores["source"] = source_score
                
                # target: individual and average
                target_scores = []
                for tgt_i, tgt in enumerate(item_sentences["targets"]):
                    score = pipe(tgt)[0]["score"]
                    target_scores.append(score)
                    item_scores[f"target_{tgt_i:02}"] = score
                item_scores["target_avg"] = np.mean(target_scores)
                
                # predictions
                prediction_scores = {}
                for prediction in item_sentences["predictions"]:
                    prediction_model, prediction_sent = prediction["model"], prediction["sentence"]
                    prediction_key = os.path.basename(prediction_model).replace(".predictions.txt", "").replace(".txt", "")
                    score = pipe(prediction_sent)[0]["score"]
                    item_scores[f"prediction_{prediction_key}"] = score
                    prediction_scores[prediction_key] = score

                # diffs (target - source)
                for tgt_i, tgt_score in enumerate(target_scores):
                    item_scores[f"diff_target_{tgt_i:02}"] = tgt_score - source_score
                item_scores["diff_target_avg"] = item_scores["target_avg"] - source_score
                for key, pred_score in prediction_scores.items():
                    item_scores[f"diff_prediction_{key}"] = pred_score - source_score

                score_table.append(item_scores)
    
        pd.DataFrame(score_table).to_csv(f"output/questionnaire/autoeval_scores.{dimension.replace(':', '-')}.group1.csv")


if __name__ == "__main__":
    autoeval()
