import argparse
import json
import os
import pandas as pd
import tiktoken

from collections import defaultdict
from pathlib import Path

from benchmark.scorer import GEvalScorer
from benchmark.external_knowledge import ExternalKnowledgeOpener
from src.agent.llm import LLM
from src.agent.utils import json_read, json_save
from src.agent.qa import QAAgent
from src.agent.visualization import VisualizationEvaluateAgent
from src.agent.curator import Curator


class QABenchmarkRunner:
    def __init__(self, dataset_dir, output_path, model, code_agent_only=False, table_head_only=False):
        self.question_files = list(Path(dataset_dir).glob("**/**/**/qa_pairs.json"))
        self.question_files = [str(p) for p in self.question_files]
        self.opener = ExternalKnowledgeOpener()
        self.output_path = output_path
        Path(self.output_path).mkdir(exist_ok=True, parents=True)
        self.qa_agent = QAAgent(output_dir=output_path, model=model, table_head_only=table_head_only)
        self.visualization_evaluator = VisualizationEvaluateAgent(model="gpt4")
        self.code_agent_only = code_agent_only
        self.dataset_dir = dataset_dir

    def _load(self, metadata, data_source, identifier):
        distributions = metadata.get("distribution")
        external_knowledge = metadata.get("external_knowledge")

        dfs = []
        dataset_paths = []

        for distribution in distributions:
            try:
                file_name = distribution["file_name"]
                dataset_path = f"{self.dataset_dir}/{data_source}/{identifier}/data/{file_name}"
                df = pd.read_csv(dataset_path)
                dfs.append(df)
                dataset_paths.append(dataset_path)
            except Exception as e:
                print(e)

        knowledge = None
        if external_knowledge is not None:
            if len(external_knowledge) > 0:
                knowledge_path = f"{self.dataset_dir}/{data_source}/{identifier}/data/{external_knowledge[0]}"
                knowledge = self.opener(dfs[0], knowledge_path)

        return dfs, dataset_paths, knowledge

    def evaluate(self):
        total = len(self.question_files)
        correct = 0
        for i, question_file in enumerate(self.question_files):
            qa_pair = json_read(str(question_file))
            questions = qa_pair["questions"]
            answers = qa_pair["answers"]

            path_splits = question_file.split("/")
            data_source = path_splits[-4]
            dataset_id = path_splits[-3]
            qid = path_splits[-2]

            code_path = "/".join(path_splits[:-1]) + "/codes"

            metadata_file = "/".join(path_splits[:-2]) + "/metadata.json"
            metadata = json_read(metadata_file)

            dfs, dataset_paths, knowledge = self._load(metadata, data_source, dataset_id)

            if self.code_agent_only:
                qa_pairs, codes = self.qa_agent.code_agent_only(dfs, dataset_paths, questions, metadata, data_source, dataset_id, qid, knowledge)
            else:   
                qa_pairs, codes = self.qa_agent(dfs, dataset_paths, questions, metadata, data_source, dataset_id, qid, knowledge)

            qa_pairs = {"qa": qa_pairs, "codes": codes}

            print(qa_pairs)

            Path(f"{self.output_path}/{data_source}/{dataset_id}/{qid}/").mkdir(exist_ok=True, parents=True)
            json_save(f"{self.output_path}/{data_source}/{dataset_id}/{qid}/results.json", qa_pairs)

            turn = 0
            matches = []
            pred_target_pairs = {}

            for i, predicted_qa in enumerate(qa_pairs["qa"]):
                if i % 2 == 1:
                    code = open(f"{code_path}/code_{turn}.py").read()
                    if isinstance(predicted_qa, list) and ".png" in predicted_qa[0]:
                        answer = answers[turn]
                        answer = [os.path.join(self.dataset_dir, a) for a in answer]
                        target_code = code
                        pred_code = codes[turn]
                        match = self.visualization_evaluator.evaluate(predicted_qa, answer, pred_code, target_code)
                        if match:
                            match = True
                        else:
                            match = False
                        matches.append(match)
                    else:
                        answer = answers[turn]
                        if answer == predicted_qa:
                            match = True
                        else:
                            match = False
                        matches.append(match)

                    pred_target_pair = {
                        "pred": predicted_qa,
                        "target": answer,
                        "match": match
                    }

                    pred_target_pairs[tentative_question] = pred_target_pair

                    turn += 1
                else:
                    tentative_question = predicted_qa

            print(matches)

            if len(matches) == sum(matches): correct += 1

            pair_path = f"{self.output_path}/{data_source}/{dataset_id}/{qid}/pred_target_pair.json"
            json_save(pair_path, pred_target_pairs)
    
    def check(self):
        generated_matches = Path(f"{self.output_path}").glob("**/**/**/pred_target_pair.json")
        multi_turn_correct = []
        detailed_correct = []

        for i, match in enumerate(generated_matches):
            match_result = json_read(str(match))

            question_number = str(match).split("/")[-2]
            correctness = True

            for question, pair in match_result.items():
                detailed_correct.append(pair["match"])
                if not pair["match"]: correctness = False

            multi_turn_correct.append(correctness)

        multi_turn_acc = sum(multi_turn_correct) / len(multi_turn_correct)
        single_acc = sum(detailed_correct) / len(detailed_correct)

        print(len(multi_turn_correct))
        print(len(detailed_correct))
        print(multi_turn_acc)
        print(single_acc)


class DIBenchmarkRunner:
    def __init__(self, dataset_dir, output_path, model, mode="report_evaluate"):
        self.dataset_dir = dataset_dir
        self.question_files = list(Path(dataset_dir).glob("**/**/insights.json"))
        self.question_files = [str(p) for p in self.question_files]
        self.opener = ExternalKnowledgeOpener()
        self.output_path = output_path
        Path(self.output_path).mkdir(exist_ok=True, parents=True)
        self.curate_agent = Curator(output_dir=output_path, model=model)

        self.gemini = LLM(method="gemini")

    @staticmethod
    def calculate_number_of_tokens( insights):
        encoding = tiktoken.get_encoding("cl100k_base")
        token_integers = encoding.encode(insights)

        return len(token_integers)

    def _load(self, metadata, data_source, identifier):
        distributions = metadata.get("distribution")
        external_knowledge = metadata.get("external_knowledge")

        dfs = []
        dataset_paths = []

        for distribution in distributions:
            try:
                file_name = distribution["file_name"]
                dataset_path = f"{self.dataset_dir}/{data_source}/{identifier}/data/{file_name}"
                try:
                    df = pd.read_csv(dataset_path)
                except Exception as e:
                    df = pd.read_csv(dataset_path, encoding="latin-1")
                dfs.append(df)
                dataset_paths.append(dataset_path)
            except Exception as e:
                print(e)

        knowledge = None
        if external_knowledge is not None:
            if len(external_knowledge) > 0:
                knowledge_path = f"{self.dataset_dir}/{data_source}/{identifier}/data/{external_knowledge[0]}"
                knowledge = self.opener(dfs[0], knowledge_path)

        return dfs, dataset_paths, knowledge
    
    def summarize(self, insights, num_tokens):
        prompt = f"""You are a top data analyst for the tabular data. Your task is to summarize the following insights generated from QA pairs related to tabular data in {num_tokens} tokens. Please summarize so that the summarization covers as diverse topics (e.g. global-to-local) as it can. Following is a list of insights.

{insights}
"""
        response = self.gemini(prompt)

        return response

    def report_evaluate(self):
        for i, question_file in enumerate(self.question_files):
            print(question_file)
            path_splits = question_file.split("/")
            data_source = path_splits[-3]
            dataset_id = path_splits[-2]

            metadata_file = "/".join(path_splits[:-1]) + "/metadata.json"
            metadata = json_read(metadata_file)

            dfs, dataset_paths, knowledge = self._load(metadata, data_source, dataset_id)

            insights, qa_pairs, summarized_insights, all_codes = self.curate_agent.generate(dfs, dataset_paths, metadata, data_source, dataset_id, external_knowledge=knowledge)
            insights = [ins.split(": ")[1] for ins in insights.split("\n") if not ins == ""]
            summarized_insights = self.summarize("\n".join(insights), 200)
            insights = {"insights": insights, "QA": qa_pairs, "summary": summarized_insights, "codes": all_codes}

            print(insights)

            Path(f"{self.output_path}/{data_source}/{dataset_id}/").mkdir(exist_ok=True, parents=True)
            json_save(f"{self.output_path}/{data_source}/{dataset_id}/report_insights.json", insights)

    def check(self):
        scorer = GEvalScorer()
        overall_scores = {}

        for i, question_file in enumerate(self.question_files):
            scores = defaultdict(list)
            print(question_file)
            path_splits = question_file.split("/")
            data_source = path_splits[-3]
            dataset_id = path_splits[-2]

            target_insights = json.load(open(str(question_file)))

            insights = json_read(f"{self.output_path}/{data_source}/{dataset_id}/report_insights.json")

            ours_insight_score, ours_insight_score_dict = scorer.compute_g_eval_o2m(insights["insights"], target_insights["insights"])
            ours_summary_score = scorer.compute_g_eval(insights["summary"], target_insights["summary"], model_name="gpt-4o", top_logprobs=5) / 10

            scores["ours_insight_score"].append(ours_insight_score)
            scores["ours_summary_score"].append(ours_summary_score)
            scores["ours_insight_score_dict"].append(ours_insight_score_dict)

            print(ours_insight_score, ours_summary_score)

            overall_scores[f"{data_source}/{dataset_id}"] = scores

            print(overall_scores)

        json_save(f"{self.output_path}/insight_scores.json", overall_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark runner")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--type", default="qa", type=str)
    parser.add_argument("--output", default="qa", type=str)
    parser.add_argument("--model", nargs='+', type=str, help="A list of items")
    parser.add_argument("--head_only", action='store_true')
    parser.add_argument("--code_only", action='store_true')
    args = parser.parse_args()
    dataset_dir = args.dataset
    benchmark_type = args.type
    output_dir = args.output
    model_name = args.model
    table_head_only = args.head_only
    code_agent_only = args.code_only

    if benchmark_type == "report_evaluate":
        output_path = f"results/{output_dir}"
        runner = DIBenchmarkRunner(dataset_dir, output_path, model_name)
        runner.report_evaluate()
        runner.check()
    elif benchmark_type == "qa_evaluate":
        output_path = f"results/{output_dir}"
        runner = QABenchmarkRunner(dataset_dir, output_path, model_name, code_agent_only=code_agent_only, table_head_only=table_head_only)
        acc = runner.evaluate()
        runner.check()
