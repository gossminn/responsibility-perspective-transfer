import re

import pandas as pd


def process_data(group="group_1"):

    if group == "group_1":
        responses_file = "questionnaire/responses/Perspective Transfer Questionnaire (group 1)_January 16, 2023_13.44.csv"
        meta_file = "output/questionnaire/questions.group1.meta.txt"
    elif group == "fewshot":
        responses_file = "questionnaire/responses/Perspective Transfer Questionnaire (few-shot prompts)_January 19, 2023_13.44.csv"
        meta_file = "output/questionnaire/questions.fewshot.meta.txt"
    else:
        raise ValueError("Invalid group")
    resp = pd.read_csv(responses_file)
    
    
    sentence_sources = []
    with open(meta_file, encoding="utf-8") as f:
        for line in f:
            source_models = []
            for sm in line.strip().split(" "):
                source_models.append(sm.replace(".prediction.txt", ""))
            # add everything to the list twice (1x perspective rating, 1x similarity rating)
            sentence_sources.extend(source_models)
            sentence_sources.extend(source_models)

    if group == "group_1":
        first_sentence_column_id = 22
        num_sentences_per_block = 7
    else:
        first_sentence_column_id = 22
        num_sentences_per_block = 5

    # build processed responses dataframe
    responses = []

    block_id = -1
    prev_block_type = None

    for i, col in enumerate(resp.columns[first_sentence_column_id:]):
        input_text = resp[col][0]
        if group == "group_1":
            cur_block_type = "similarity" if input_text.startswith("(B.") else "perspective"
        else:
            cur_block_type = "similarity" if input_text.startswith("(A)") else "perspective"
        
        if cur_block_type == "perspective" and cur_block_type != prev_block_type:
            block_id += 1
        prev_block_type = cur_block_type

        # get the question text, remove question number if necessary
        if cur_block_type == "similarity": 
            if group == "group_1":
                input_text = re.sub(r"^\(B\.\d\) ", "", input_text)
            else:
                input_text = re.sub(r".*\(B\) ", "", input_text.split("\n")[-1])
        
        # model
        model = sentence_sources.pop(0)

        answers = resp[col][2:].to_list()
        for ans_id, ans in enumerate(answers):
            if ans == "-99":
                score = 0 if cur_block_type == "perspective" else 5
            else:
                score = int(ans)
            responses.append({
                "block_id": block_id,
                "user_id": ans_id,
                "model": model,
                "category": cur_block_type,
                "sentence": input_text,
                "score": score
            })
    
    responses_df = pd.DataFrame(responses)
    responses_df.to_csv(f"output/questionnaire/responses.processed.{group}.csv")

    responses_df.groupby("model").agg({"score": "mean"})

if __name__ == "__main__":
    process_data("group_1")
    process_data("fewshot")