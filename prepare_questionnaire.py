import argparse
from collections import defaultdict
import glob
import json
import os
import random

random.seed(1996)


NUM_SENTENCES = 50
SENTENCES_PER_GROUP = 50

PROMPT_BLAME = """
Per ciascuna frase usa il tachimetro per valutare quanto si sofferma, secondo te, sulla colpa dell’assassino
""".strip()

PROMPT_SIMILARITY = """
Usa il "termometro" per indicare quanto è probabile che la frase (A) e ciascuna delle frasi (B.1) - (B.7) descrivano gli stessi fatti.
""".strip()

def generate():
    sources = []
    with open("output/hum_eval_set_pairs.jsonl", encoding="utf-8") as f:
        prev_source = None
        for line in f:
            pair_data = json.loads(line)
            source = pair_data["sentence_low"]
            if source != prev_source:
                sources.append(source)
            prev_source = source
    print(len(sources))

    predictions = [[] for _ in sources]
    prediction_files = glob.glob("questionnaire/sentences_for_questionnaire_group1/*.txt")
    print(prediction_files)
    for pfi, prediction_file in enumerate(prediction_files):
        print(pfi + 1)
        with open(prediction_file, encoding="utf-8") as f:
            prev_line = None
            line_idx = 0
            for line in f:
                line = line.strip()
                if line != prev_line:
                    predictions[line_idx].append(line)
                    line_idx += 1
                prev_line = line
    
    print(len(predictions[0]))

    num_versions = NUM_SENTENCES // SENTENCES_PER_GROUP
    version_files = [f"output/questionnaire/questions.v{i+1}.txt" for i in range(num_versions)]
    version_meta_files = [vf.replace(".txt", ".meta.txt") for vf in version_files]
    version_sents_files = [vf.replace(".txt", ".sents.jsonl") for vf in version_files]

    sampled_keys = random.sample(list(range(len(predictions))), k=NUM_SENTENCES)

    for vf, mf, sf in zip(version_files, version_meta_files, version_sents_files):
        with open(vf, "w", encoding="utf-8") as f, open(mf, "w", encoding="utf-8") as f_meta, open(sf, "w", encoding="utf-8") as f_sents:
            f.write("[[AdvancedFormat]]\n")
            for _ in range(SENTENCES_PER_GROUP):
                sentence_key = sampled_keys.pop()        
                source_sentence = sources[sentence_key]

                # generate random order for predictions and write them to meta file
                prediction_idx = list(range(len(predictions[sentence_key])))
                random.shuffle(prediction_idx)
                for pix in prediction_idx:
                    f_meta.write(os.path.basename(prediction_files[pix]) + " ")
                f_meta.write("\n")

                target_sentences = []
                f.write("\n[[Block]]\n")
                f.write(f"\n[[Question:Text]]\n{PROMPT_BLAME}\n")
                for pix in prediction_idx:
                    sentence = predictions[sentence_key][pix]
                    target_sentences.append({"model": prediction_files[pix], "sentence": sentence})
                    f.write(f"\n[[Question:MC:Select]]\n{sentence}\n[[Choices]]\nSPEEDO\nSPEEDO\n")
                f.write(f"\n[[Question:Text]]\n{PROMPT_SIMILARITY}\n(A) {source_sentence}\n")
                for i, pix in enumerate(prediction_idx):
                    sentence = predictions[sentence_key][pix]
                    f.write(f"\n[[Question:MC:Select]]\n(B.{i+1}) {sentence}\n[[Choices]]\nTHERMO\nTHERMO\n")

                sentence_info = {
                    "source": source_sentence,
                    "targets": target_sentences
                }
                f_sents.write(json.dumps(sentence_info) + os.linesep)

def format():

    with open("questionnaire/graded_scale_template.qsf", encoding="utf-8") as f:
        graded_scales_tmpl = json.load(f)

    with open("output/questionnaire/questions.qsf", encoding="utf-8") as f:
        questionnaire = json.load(f)
        for se in questionnaire["SurveyElements"]:
            if se.get("Payload") is None:
                continue
            if se.get("Element") != "SQ":
                continue
            if se["Payload"].get("QuestionType", "") == "DB":
                se["Payload"]["QuestionText"] = (
                    "<span style=\"font-size:20px;\">"
                    + "<strong>" # boldface 
                    + se["Payload"]["QuestionText"].replace("fatti.\n(A)", "fatti. <br><br><em>(A)") + "</em>" # newline and italics for source sentence 
                    + "</strong>"
                    + "</span>"
                    
                )
            if se["Payload"].get("QuestionType", "") == "MC":
                payload = {}
                for key in graded_scales_tmpl["Payload"]:
                    if key == "Scale":
                        if se["Payload"]["Choices"]["1"]["Display"] == "SPEEDO":
                            payload[key] = "TenGauge"
                        else:
                            payload[key] = "Thermometer"
                    elif graded_scales_tmpl["Payload"][key] == "#####":
                        payload[key] = se["Payload"][key]
                    else:
                        payload[key] = graded_scales_tmpl["Payload"][key]
                payload["DataExportTag"] = se["Payload"]["DataExportTag"].replace("QID", "Q")
                se["Payload"] = payload

    with open("output/questionnaire/questions.fmt.qsf", "w", encoding="utf-8") as f:
        json.dump(questionnaire, f, indent=4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["generate", "format"])
    args = ap.parse_args()

    if args.action == "generate":
        generate()
    else:
        format()