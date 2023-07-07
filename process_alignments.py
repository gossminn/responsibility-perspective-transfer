import pandas as pd
import json

from Levenshtein import distance as lv_distance


def get_original_sentence(sentence_from_alignments, events_to_sentences, event_id, doc_id):
    potential_sentences = [
        (sent, lv_distance(sentence_from_alignments, sent["sentence"]))
        for sent in events_to_sentences[str(event_id)]
        if sent["document_id"] == doc_id
    ]

    if not potential_sentences:
        # raise ValueError(f"No sentences found for event={event_id}, document={doc_id}")
        print(f"WARNING: No sentences found for event={event_id}, document={doc_id}")
        return {"sentence": None}

    output = min(potential_sentences, key=lambda t: t[1])[0]
    print(f"Matched:\n\tX: {sentence_from_alignments}\n\tY:{output['sentence']}\n")
    return output


def process_alignments():

    # with open("output/preproc/gold.grouped_by_event.json", encoding="utf-8") as f:
    #     events_to_sentences = json.load(f)
    with open("output/preproc/gold.grouped_by_event.json", encoding="utf-8") as f_gold,\
         open("output/preproc/silver.grouped_by_event.json", encoding="utf-8") as f_silver:
        events_to_sentences = json.load(f_gold) | json.load(f_silver)

    test_sents = [
        sentence["sentence"] 
        for event_sents in events_to_sentences.values()
        for sentence in event_sents
        if sentence["split"] == "test"
    ]   

    sentences = (
        pd.read_excel("data/alignment-annotations/final_aligned_version.xlsx")
        .dropna()
        .convert_dtypes({"EVENT": int, "DOCUMENT_ID": int})
        .rename(columns={"EVENT": "event", "GROUP": "group", "SENTENCE": "sentence", "DOCUMENT_ID": "doc_id", "B_PERCEPTION": "blame_perc"})
        .pipe(lambda df: df if "group" in df.columns else df.assign(group=lambda _: 1))  # assign default group
        # string cleaning
        .assign(
            blame_perc=lambda df: df["blame_perc"].apply(lambda x: x.strip()),
            sentence=lambda df: df.apply(lambda row: get_original_sentence(row["sentence"], events_to_sentences, row["event"], row["doc_id"])["sentence"], axis=1)
        )
        .dropna()
        .assign(
            split=lambda df: df["sentence"].apply(lambda sent: "test" if sent in test_sents else "train+dev")
        )
        .groupby(["event", "group"])
        .agg({"event": "first", "group": "first", "sentence": list, "doc_id": list, "blame_perc": list, "split": list})
    )
    
    # sentences.to_json("output/align/alignments.gold-silver.bm.2022-10-14.jsonl", orient="records", lines=True)

if __name__ == "__main__":
    process_alignments()