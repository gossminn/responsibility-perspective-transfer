"""
Prepare an initial test set (n=200)
"""

import os
import pandas as pd


def prepare_test_set():
    sentences_to_scores = map_sentences_to_scores()
    pairs_with_distance = compile_pairs(sentences_to_scores)
    shuffled_and_grouped = shuffle_and_group(pairs_with_distance)
    first_250_idx = shuffled_and_grouped[shuffled_and_grouped["size"].cumsum() > 250].index[0]
    next_500_idx = shuffled_and_grouped[shuffled_and_grouped["size"].cumsum() > 750].index[0]
    next_next_500_idx = shuffled_and_grouped[shuffled_and_grouped["size"].cumsum() > 1250].index[0]
    test_events = shuffled_and_grouped.iloc[: first_250_idx + 1]["event"].to_list()
    hum_eval_events = shuffled_and_grouped.iloc[first_250_idx + 1 : next_500_idx + 1]["event"].to_list()
    hum_eval_2_events = shuffled_and_grouped.iloc[next_500_idx + 1 : next_next_500_idx + 1]["event"].to_list()
    test_set = pairs_with_distance[pairs_with_distance["event"].isin(test_events)]
    hum_eval_set = pairs_with_distance[pairs_with_distance["event"].isin(hum_eval_events)]
    hum_eval_2_set = pairs_with_distance[pairs_with_distance["event"].isin(hum_eval_2_events)]
    test_set.to_json("output/test_set_pairs.jsonl", orient="records", lines=True)
    hum_eval_set.to_json("output/hum_eval_set_pairs.jsonl", orient="records", lines=True)
    hum_eval_2_set.to_json("output/hum_eval_set_2_pairs.jsonl", orient="records", lines=True)


def map_sentences_to_scores():
    sentences_to_scores = {}
    for sentence_file in [
        "output/preproc/silver.full.filtered.jsonl",
        "output/preproc/gold.full.jsonl"]:

        sentences_df = pd.read_json(sentence_file, orient="records", lines=True)
        for entry in sentences_df.itertuples():
            sentence = entry.sentence
            perception_blame_murderer = entry.perception["B:suspect"]
            sentences_to_scores[sentence] = perception_blame_murderer
    return sentences_to_scores


def compile_pairs(sentences_to_scores):
    pairs_with_distance = []
    mislabeled_low = set()
    mislabeled_high = set()
    align_pairs = pd.read_json("output/align/alignments.gold-silver.bm.2022-10-14.combinations.jsonl", orient="records", lines=True)
    for pair in align_pairs.itertuples():
        sentence_low = pair.sentence_low
        sentence_high = pair.sentence_high
        score_low = sentences_to_scores[sentence_low]
        score_high = sentences_to_scores[sentence_high]
        diff = score_high - score_low

        # for some reason, a few samples got mislabeled (during the alignment process?)
        if score_low > 0:
            mislabeled_low.add(sentence_low)
            continue
        if score_high < 0:
            mislabeled_high.add(sentence_high)
            continue

        pairs_with_distance.append({
            "event": pair.event,
            "sentence_low": sentence_low,
            "sentence_high": sentence_high,
            "score_low": score_low,
            "score_high": score_high,
            "diff": diff
        })
    return pd.DataFrame(pairs_with_distance)


def shuffle_and_group(pairs_with_distance):
    sorted_diff_events = (
        pairs_with_distance
        .groupby("event")
        .agg(**{"diff": ("diff", "mean"), "size": ("diff", "size")})
        .sample(frac=1, random_state=1996)
        .reset_index()
    )
    return sorted_diff_events


if __name__ == "__main__":
    prepare_test_set()
