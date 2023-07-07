"""
Extract few-shot training + evaluation examples for prompt engineering experiment
"""
import os
import pandas as pd

RANDOM_SEED = 14_10_58


def prepare_gpt3():
    combinations = pd.read_json("output/align/alignments.gold-silver.bm.2022-10-14.uniq_combinations.jsonl", orient="records", lines=True, encoding="utf-8")
    combinations_train = combinations[combinations["split"] == "train+dev"]

    samples = combinations_train.sample(n=30, random_state=RANDOM_SEED)
    train_samples = samples.iloc[:10]
    eval_samples = samples.iloc[10:]

    for sample_df, out_name in [(train_samples, "training.txt"), (eval_samples, "evaluation.txt")]:
        with open(f"output/for_gpt3/{out_name}", "w", encoding="utf-8") as f:
            for _, row in sample_df.iterrows():
                f.write(row["sentence_high"] + "*****" + row["sentence_low"] + os.linesep)


if __name__ == "__main__":
    prepare_gpt3()
