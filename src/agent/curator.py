import ast
import pandas as pd

from pathlib import Path

from src.agent.qa import QAAgent
from src.agent.insight import InsightAgent, InsightGraph
from src.agent.plan import Planner


class Curator:
    def __init__(self, output_dir, model="gpt4", table_head_only=False):
        self.output_dir = output_dir
        root_number = 3
        ratio = 1
        self.insight_agent = InsightAgent(output_dir=self.output_dir, model=model[0], root_number=root_number, ratio=ratio)
        self.planner = Planner(output_dir=self.output_dir, model=model[0], root_number=root_number, ratio=ratio)
        self.qa_agent = QAAgent(output_dir=self.output_dir, model=model, table_head_only=table_head_only)

        self.insight_graph = InsightGraph(output_dir)
        self.iterations = 4

    def _add_qa(self, questions, answers, iteration, references=None):
        replaced_questions = []
        if iteration == 0:
            for index, (question, answer) in enumerate(zip(questions, answers)):
                node_id = f"Q_{iteration}_{index}"
                self.insight_graph.add_qa_node(iteration, index, question, answer)
                replaced_question = question.replace(f"Q{index}", node_id)
                replaced_questions.append(replaced_question)
        else:
            for index, (question, answer, ref) in enumerate(zip(questions, answers, references)):
                node_id = f"Q_{iteration}_{index}"
                self.insight_graph.add_qa_node(iteration, index, question, answer, insight_refs=ref)
                replaced_question = question.replace(f"Q{index}", node_id)
                replaced_questions.append(replaced_question)

        return replaced_questions

    def _add_insight(self, insights, iteration, chains):
        for i, (insight, chain) in enumerate(zip(insights, chains)):
            self.insight_graph.add_insight_node(iteration, i, insight, chain)

    def generate(self, dfs, dataset_paths, metadata, data_source, dataset_id, external_knowledge=None):
        previous_insights = ""
        previous_qa_pairs = []
        all_codes = []

        self.insight_graph = InsightGraph(self.output_dir)

        for iteration in range(self.iterations):
            if iteration == 0:
                questions = self.planner(dfs, metadata, data_source, external_knowledge)
                references = None
            else:
                questions = self.planner.followup(dfs, metadata, data_source, external_knowledge, previous_qa_pairs, previous_insights, iteration)
                replaced_questions, references = [], []
                for q_dict in questions:
                    replaced_questions.append(q_dict["question"])
                    if not isinstance(q_dict["reference"], list):
                        if q_dict["reference"].startswith("["):
                            q_dict["reference"] = ast.literal_eval(q_dict["reference"])
                        elif isinstance(q_dict["reference"], str):
                            q_dict["reference"] = q_dict["reference"].split(", ")
                    references.append(q_dict["reference"])
                questions = replaced_questions

            qa_pairs, codes = self.qa_agent(dfs, dataset_paths, questions, metadata, data_source, dataset_id, 0, external_knowledge=external_knowledge)
            all_codes += codes

            replaced_qa_pairs = []
            questions, answers = [], []
            q_count = 0
            for j, qa_pair in enumerate(qa_pairs):
                if j % 2 == 0:
                    qa_pair = f"Q_{iteration}_{q_count}: " + qa_pair.lstrip(" ")
                    questions.append(qa_pair)
                    q_count += 1
                else: answers.append(qa_pair)
                replaced_qa_pairs.append(qa_pair)

            _ = self._add_qa(questions, answers, iteration, references)

            previous_qa_pairs += replaced_qa_pairs

            insights = self.insight_agent(dfs, metadata, data_source, external_knowledge=external_knowledge, previous_qa_pairs=previous_qa_pairs, insights=None, iteration=iteration)

            generated_insights = []
            insight_references = []

            for i, i_dict in enumerate(insights):
                replaced_refs = []
                if not isinstance(i_dict["reference"], list):
                    if i_dict["reference"].startswith("["):
                        i_dict["reference"] = ast.literal_eval(i_dict["reference"])
                    elif isinstance(i_dict["reference"], str):
                        i_dict["reference"] = i_dict["reference"].split(", ")
                for ref in i_dict["reference"]:
                    if ref.startswith("Q"):
                        replaced_refs.append(ref)
                insight = i_dict["insight"]
                previous_insights += f"I_{iteration}_{i} (generated from {replaced_refs}): {insight}\n"

                generated_insights.append(insight)
                insight_references.append(i_dict["reference"])

            self._add_insight(generated_insights, iteration, insight_references)

        self.insight_graph.serialize(data_source, dataset_id)

        summarized_insights = self.insight_agent.summarize(previous_insights)
        return previous_insights, previous_qa_pairs, summarized_insights, all_codes