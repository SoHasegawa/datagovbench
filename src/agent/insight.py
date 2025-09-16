import networkx as nx
import json
import matplotlib.pyplot as plt

from pathlib import Path
from networkx.readwrite import json_graph

from src.agent.sequence import sequence_qa_pairs, sequence_external_knowledge, sequence_tables
from src.agent.llm import LLM



class InsightGraph:
    def __init__(self, output_dir):
        self.graph = nx.Graph()
        self.insight_records = {}
        self.output_dir = output_dir

    def add_qa_node(self, iteration, index, question, answer, insight_refs=None):
        node_id = f"Q_{iteration}_{index}"
        self.graph.add_nodes_from([(node_id, {"question": question, "answer": answer, "type": "QA"})])

        if insight_refs is not None:
            for insight_ref in insight_refs:
                self.graph.add_edge(insight_ref, node_id)

    def add_insight_node(self, iteration, index, insight, questions):
        node_id = f"I_{iteration}_{index}"
        self.graph.add_nodes_from([(node_id, {"insight": insight, "type": "insight"})])

        for question_id in questions:
            self.graph.add_edge(question_id, node_id)

    def serialize(self, data_source, dataset_id):
        data = json_graph.node_link_data(self.graph)

        Path(f"{self.output_dir}/{data_source}/{dataset_id}/").mkdir(exist_ok=True, parents=True)
        with open(f"{self.output_dir}/{data_source}/{dataset_id}/insight_graph.json", 'w') as f:
            json.dump(data, f, indent=2)


class InsightAgent:
    def __init__(self, output_dir, model="gpt4", root_number=5, ratio=1):
        if isinstance(model, str):
            self.llm = LLM(method=model)
        else:
            self.llm = model
        self.output_dir = f"{output_dir}/insights"

        self.root_number = root_number
        self.ratio = ratio

    def construct_prompt(self, dfs, metadata, data_source, external_knowledges=None, previous_qa_pairs=None, insights=None, iteration=0):
        output_format = [{"reference": "List of question numbers (e.g. ['Q_0_0', 'Q_1_0'])", "insight": "Insight text"}, {"reference": "...", "insight": "..."}]
        output_format = str(output_format)
        dataset_title = metadata["dataset_title"]
        dataset_description = metadata["dataset_description"]
        publisher = metadata["publisher"]

        table_information = sequence_tables(dfs, metadata)

        if external_knowledges is not None:
            knowledge = sequence_external_knowledge(external_knowledges)

        if len(previous_qa_pairs) > 0:
            pairs, image_paths = sequence_qa_pairs(previous_qa_pairs)

        number_of_insights = self.root_number * (self.ratio ** iteration)

        prompt = f"""You are a top data analyst for the tabular data. Your task is to generate insights that can be read from table information QA pairs related to the table. Insights should follow the following points.
- Insights are text-based, and should include factual and informative information that attract readers. Also, they are expected to invoke non-trivial realizations for humans.
- Insight should be generated from QA pairs, and please include which QA pairs contribute to the insight by referring to the corresponding question numbers.
- Write down {number_of_insights} insights, and the output format is {output_format} in the strict JSON format without markdown formatting and indentation. Do not add additional texts or redundant information.
- Single insight can be produced from one or multiple QA pairs. It is encouraged to aggregate multiple QA pairs to generate single insight.
- Avoid logical leap under many uncertain assumptions.
- Avoid duplications of insights by referring to the insights generated so far.

Following is QA pairs and table information.

## QA Pairs so far
{pairs}

## Insights so far
{insights}

## Table Information
Data Origin: {data_source}
Data publisher: {publisher}
Data source: {dataset_title}
Data source description: {dataset_description}
"""
        
        for t_info in table_information:
            prompt += f"{t_info}\n\n"

        if external_knowledges is not None:
            prompt += "Following is the external knowledge on the tabular information. Please use the knowledge to make the ID-like or ambiguous words clear.\n"
            prompt += knowledge
            
        return prompt, image_paths
    
    def _save_insights(self, insights, iteration, summarized=False):
        if summarized:
            output_path = f"{self.output_dir}/summarized_insights.txt"

            with open(output_path, 'w') as f:
                if insights is None: insights = ""
                f.write(insights)
        else:
            Path(f"{self.output_dir}/iter{iteration}").mkdir(exist_ok=True, parents=True)
            output_path = f"{self.output_dir}/iter{iteration}/insights.json"

            with open(output_path, 'w') as f:
                json.dump(insights, f)

    def summarize(self, insights):
        prompt = f"""You are a top data analyst for the tabular data. Your task is to summarize the following insights generated from QA pairs related to tabular data. Please summarize so that the summarization covers as diverse topics (e.g. global-to-local) as it can. Following is a list of insights.

{insights}
"""
        response = self.llm(prompt)
        self._save_insights(response, 0, summarized=True)

        return response
    
    def __call__(self, dfs, metadata, data_source, external_knowledge=None, previous_qa_pairs=None, insights=None, iteration=0, temperature=0.0):
        prompt, image_paths = self.construct_prompt(dfs, metadata, data_source, external_knowledge, previous_qa_pairs, insights, iteration)
        if len(image_paths) == 0: image_paths = None
        insight = self.llm.generate_format(prompt, image_paths=image_paths, temperature=temperature, format="json")

        self._save_insights(insight, iteration, summarized=False)

        return insight
