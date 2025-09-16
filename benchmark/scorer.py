import time
import openai
import re
import os
import numpy as np

from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
from collections import defaultdict


G_EVAL_BASIC_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Provided Answer:
{answer}

Ground Truth Answer:
{gt_answer}

Follow these instructions when writing your response:
* On a scale of 1-10, provide a numerical rating for how close the provided answer is to the ground truth answer, with 10 denoting that the provided answer is the same as ground truth answer.
* Your response should contain only the numerical rating. DONOT include anything else like the provided answer, the ground truth answer, or an explanation of your rating scale in your response.
* Wrap your numerical rating inside <rating></rating> tags.
* Check very carefully before answering.
* Follow the output format as shown in the example below:
Example response:
<rating>7</rating>

### Response:

"""

G_EVAL_BINARY_SYSTEM_MESSAGE = """You are a high school teacher evaluating student responses to a question. You are tasked with grading the response based on how well it answers the question. You are to provide a numerical rating for how well the provided response matches the ground truth answer."""

G_EVAL_BASIC_SYSTEM_MESSAGE = """You are a high school teacher evaluating student responses to a question. You are tasked with grading the response based on how well it answers the question. You are to provide a numerical rating for how well the response answers the question based on the ground truth answer."""


G_EVAL_BINARY_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Provided answer:
{answer}

GT Answer:
{gt_answer}

On a scale of 1-10, provide a numerical rating for how close the provided answer is to the ground truth answer, with 10 denoting that the provided answer is the the same as ground truth answer. The response should contain only the numerical rating.\
    
Check very carefully before answering.

### Response:
"""

G_EVAL_SYSTEM_MESSAGE = """You are a a high school teacher evaluating student responses to a question. You are tasked with grading the response based on how well it answers the question. You are to provide a numerical rating for how well the response answers the question based on the ground truth answer."""


G_EVAL_M2M_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Predicted Answers:
{pred_list}

Grouth Truth Answers:
{gt_list}

For each ground truth answer above, provide the index of the most appropriate predicted answer (1-indexed).
Each line must contain a single integer value denoting the id of the matched prediction.
If there is no appropriate prediction for a ground truth answer, write -1.
Check very carefully before answering.

### Response:
"""

G_EVAL_M2M_SYSTEM_MESSAGE = "You are a high school teacher evaluating student responses to some questions. Before scoring their answers, you need to first match each ground truth answer with the most appropriate answer provided by the student."
    

class GEvalScorer:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version="DUMMY"
        )

    def compute_g_eval(self, answer, gt_answer, model_name="gpt-4o", top_logprobs=False):
        return self.compute_llm_eval(answer, gt_answer, model_name, top_logprobs)

    def compute_llm_eval(self, answer, gt_answer, model_name="gpt-4o", top_logprobs=None):
        template, system_message = G_EVAL_BASIC_TEMPLATE, G_EVAL_BASIC_SYSTEM_MESSAGE

        prompt = template.format(answer=answer, gt_answer=gt_answer)
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_message,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0,
                    max_tokens=50,
                    top_p=1,
                    logprobs=bool(top_logprobs),
                    top_logprobs=top_logprobs,
                )
                if not top_logprobs:
                    score = response.choices[0].message.content
                else:
                    # get the index in response where we have the rating
                    rating_str = re.findall(
                        r"<rating>(\d+)</rating>", response.choices[0].message.content
                    )[0]
                    tokens = [o.token for o in response.choices[0].logprobs.content]
                    rating_idx_in_response = tokens.index(rating_str)
                    response = (
                        response.choices[0]
                        .logprobs.content[rating_idx_in_response]
                        .top_logprobs
                    )
                    # convert logprobs to probs
                    probs = [np.exp(obj.logprob) for obj in response]
                    # renormalize probs to sum to 1
                    probs = [obj / sum(probs) for obj in probs]
                    ratings = [
                        float(obj.token) if obj.token.isdigit() else 0 for obj in response
                    ]
                    # final score
                    score = sum([a * b for a, b in zip(ratings, probs)])
                try:
                    score = float(score)
                except ValueError:
                    score = float(score.splitlines()[0])
                except:
                    score = 0
                return score
            except openai.RateLimitError as e:
                print("RateLimitError, Sleeping for 100 seconds...")
                time.sleep(100)
            except openai.APIError as e:
                print(f"APIError, {e}\nSleeping for 100 seconds...")
                time.sleep(100)
            except Exception as e:
                print(f"{e}, Sleeping for 100 seconds...")

    def compute_g_eval_m2m(self, pred_insights, gt_insights, model_name="gpt-4o", top_logprobs=None):
        """Does many-to-many matching of provided and gt insights"""
        template = G_EVAL_M2M_TEMPLATE
        pred_insights_formatted = "\n".join(
            [f"{idx+1}. {a}" for idx, a in enumerate(pred_insights)]
        )
        gt_answers_formatted = "\n".join(
            [f"{idx+1}. {a}" for idx, a in enumerate(gt_insights)]
        )
        prompt = template.format(
            pred_list=pred_insights_formatted, gt_list=gt_answers_formatted
        )
        fail_count = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": G_EVAL_M2M_SYSTEM_MESSAGE,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0,
                    max_tokens=50,
                    top_p=1,
                    logprobs=bool(top_logprobs),
                    top_logprobs=top_logprobs,
                )
                matched_responses = []
                for line in response.choices[0].message.content.splitlines():
                    if line.strip().isdigit():
                        matched_responses.append(int(line.strip()))
                    else:  # try to capture 1. -1 type outputs
                        matched_responses.append(
                            int(re.sub(r"\d\.\s(.+)", r"\1", line).strip())
                        )
                scores_dict = []
                for id, mid in enumerate(matched_responses):
                    mid = mid - 1 if mid > 0 else np.random.choice(len(pred_insights))
                    score = (
                        self.compute_g_eval(
                            pred_insights[mid],
                            gt_insights[id],
                            model_name,
                            top_logprobs,
                        )
                        / 10.0
                    )
                    scores_dict.append(
                        {
                            "pred_insight": pred_insights[mid],
                            "gt_insight": gt_insights[id],
                            "score": score,
                        }
                    )
                score = np.mean([score["score"] for score in scores_dict])
                return score, scores_dict
            except openai.RateLimitError as e:
                print("RateLimitError, Sleeping for 100 seconds...")
                time.sleep(100)
            except openai.APIError as e:
                print(f"APIError, {e}\nSleeping for 100 seconds...")
                time.sleep(100)
            except Exception as e:
                print(f"Error occured: {e}, Retrying")
                if fail_count <= 5:
                    fail_count += 1
                    continue
                print("Retries exhausted, returning random match G-Eval results")
                # return random matching results
                scores_dict = []
                for id in range(len(gt_insights)):
                    mid = np.random.choice(len(pred_insights))
                    score = (
                        self.compute_g_eval(
                            pred_insights[mid],
                            gt_insights[id],
                            top_logprobs,
                        )
                        / 10.0
                    )
                    scores_dict.append(
                        {
                            "pred_insight": pred_insights[mid],
                            "gt_insight": gt_insights[id],
                            "score": score,
                        }
                    )
                score = np.mean([score["score"] for score in scores_dict])
                return score, scores_dict

    def compute_g_eval_o2m(self, pred_insights, gt_insights, return_scores=False):
        """
        Compute the G-Eval score for a list of predictions and ground truths.

        Args:
        -----
        pred_insights (List[str]): The list of predicted insights.
        gt_insights (List[str]): The list of ground truth insights.

        Returns:
        --------
        score (float): The G-Eval score.
        """
        # Compute the G-Eval (many-to-many version) score for each prediction and ground truth pair
        # Compute the G-Eval (one-to-many version) score for each prediction and ground truth pair
        scores_list = defaultdict(list)

        pbar = tqdm(enumerate(gt_insights), leave=False, total=len(gt_insights))
        for idx, insight in pbar:
            for pred_insight in pred_insights:
                scores_list[idx].append(
                    self.compute_g_eval(pred_insight, insight, top_logprobs=5) / 10.0
                )
        score_dict = []
        for gt_id in scores_list:
            best_pred_id = np.argmax(scores_list[gt_id])
            score_dict.append(
                {
                    "pred_insight": pred_insights[best_pred_id],
                    "gt_insight": gt_insights[gt_id],
                    "score": scores_list[gt_id][best_pred_id],
                }
            )
        score = np.mean([score["score"] for score in score_dict])
        return score, score_dict
