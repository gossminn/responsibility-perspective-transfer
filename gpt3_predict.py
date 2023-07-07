import os
import json
import time

import openai
# from pychatgpt import Chat

SECRETS_FILE = "secrets.json"
# SECRETS_FILE = "secrets.veerle.json"


with open(SECRETS_FILE) as f:
    secrets = json.load(f)
    openai.api_key = secrets["openai_api_key"]


def gpt3_predict(mode="0SLH", openai_model="text-davinci-002"):
    assert mode in ["0SLH", "FSLH", "CFSLH", "CFSLH-B"]

    with open(f"data/gpt3_prompts/prompt_{mode}.txt", encoding="utf-8") as f:
        prompt = f.read()

    if mode == "FSLH":
        with open("output/for_gpt3/training.txt", encoding="utf-8") as f:
            for line in f:
                sent_low, sent_high = line.strip().split("*****")
                prompt = prompt.replace(
                    "[[EXAMPLES]]", f"{sent_low} ***** {sent_high}\n[[EXAMPLES]]"
                )
            prompt = prompt.replace("[[EXAMPLES]]", "")

    # responses_file = f"output/gpt3_predictions/gpt3_{mode}_test.responses.jsonl"
    # responses_file = f"output/gpt3_predictions/gpt3_{mode}_hum_eval.responses.jsonl"
    responses_file = f"output/gpt3_predictions/gpt3_{mode}_hum_eval_2.responses.jsonl"
    if not os.path.exists(responses_file):       
        with open(responses_file, "w", encoding="utf-8") as f_out:
            # with open("output/test_set_pairs.jsonl", encoding="utf-8") as f_test:
            # with open("output/hum_eval_set_pairs.jsonl", encoding="utf-8") as f_test:
            with open("output/hum_eval_set_2_pairs.jsonl", encoding="utf-8") as f_test:
                prev_test_sample = None
                prev_response = None
                for line in f_test:
                    sample_data = json.loads(line)
                    test_sample = sample_data["sentence_low"]

                    if test_sample != prev_test_sample:
                        prompt_w_sample = prompt.replace("[[TEST_SAMPLE]]", test_sample)
                        print(prompt_w_sample)
                        print()

                        if openai_model == "chatgpt":
                            chat = Chat(email=secrets["email"], password=secrets["password"])
                            attempts = 0
                            generation = chat.ask(prompt_w_sample)
                            while generation == "Error":
                                if attempts >= 3:
                                    raise ConnectionError("Couldn't get a valid response from ChatGPT")
                                attempts += 1
                                generation = chat.ask(prompt_w_sample)
                            response = {"model": "chatgpt", "choices": [{"text": generation}]}
                        else:
                            response = openai.Completion.create(
                                model=openai_model,
                                prompt=prompt_w_sample,
                                temperature=0.7,
                                max_tokens=256,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0,
                                best_of=1,
                            )
                            time.sleep(2)
                    else:
                        response = prev_response
                    f_out.write(json.dumps(response) + os.linesep)

                    prev_test_sample = test_sample
                    prev_response = response

    # final_predictions_file = f"output/gpt3_predictions/gpt3_{mode}_test.predictions.txt"
    # final_predictions_file = f"output/gpt3_predictions/gpt3_{mode}_hum_eval.predictions.txt"
    final_predictions_file = f"output/gpt3_predictions/gpt3_{mode}_hum_eval_2.predictions.txt"
    with open(responses_file, encoding="utf-8") as f_resp:
        with open(final_predictions_file, "w", encoding="utf-8") as f_pred:
            for line in f_resp:
                resp_data = json.loads(line)
                predicted_text = (
                    resp_data["choices"][0]["text"]
                    .strip()
                    .split("*****")[0]
                    .split("\n")[0]
                )
                f_pred.write(predicted_text + os.linesep)

if __name__ == "__main__":
    gpt3_predict(mode="0SLH")
    gpt3_predict(mode="FSLH")
    gpt3_predict(mode="CFSLH")
    gpt3_predict(mode="CFSLH-B")
    # gpt3_predict(mode="CFSLH-B", openai_model="text-davinci-003")
    # gpt3_predict(mode="CFSLH-B", openai_model="chatgpt")
