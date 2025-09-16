import pandas as pd
import re
import io
import traceback
import contextlib

from pathlib import Path

from src.agent.llm import LLM
from src.agent.utils import json_save, json_read

from src.agent.visualization import VisualizationAgent
from src.agent.error_handling import ErrorHandlingAgent
from src.agent.code import CodeAgent
from src.agent.table import TableReflectionAgent


class QAAgent:
    def __init__(self, output_dir, model=["gpt4"], table_head_only=False):
        if len(model) == 1:
            self.local_oss = False
            self.llm = LLM(method=model[0], multimodal=False, system_prompt_enable=True)
            self.mllm = LLM(method=model[0], multimodal=False, system_prompt_enable=True)
        else:
            self.local_oss = True
            self.llm = LLM(method=model[0], multimodal=False, system_prompt_enable=True)
            self.mllm = LLM(method=model[1], multimodal=True, system_prompt_enable=True)

        self.visualization_agent = VisualizationAgent(self.mllm)
        self.error_handling_agent = ErrorHandlingAgent(self.llm)
        self.reflection_agent = TableReflectionAgent(self.llm, local_llm=self.local_oss, head_only=table_head_only)
        self.code_agent = CodeAgent(self.llm, local_llm=self.local_oss, head_only=table_head_only)
        self.output_dir = f"{output_dir}/answers"
        self.turn = 0

    def _execute(self, code):
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()

        with contextlib.redirect_stdout(output_buffer):
            with contextlib.redirect_stderr(error_buffer):
                try:
                    exec(code, {})
                except Exception as e:
                    traceback.print_exc()

        stdout_output = output_buffer.getvalue()
        stderr_output = error_buffer.getvalue()

        return stdout_output, stderr_output

    def fill_code(self, code, dataset_paths, output_dir, data_source, dataset_id, question_id, turn):
        for index, dataset_path in enumerate(dataset_paths):
            code = code.replace(f"data_{index+1}.csv", dataset_path)
        savefig_pattern = r"plt\.savefig\(\s*f?'"
        savefig_matches = re.findall(savefig_pattern, code)
        
        splits = output_dir.split("/")
        if len(savefig_matches) > 0:
            output_image_dir = "/".join(splits) + f"/{data_source}/{dataset_id}/rank{question_id}/turn{turn}/"
            Path(output_image_dir).mkdir(parents=True, exist_ok=True)
            output_sentence = f"plt.savefig(f'{output_image_dir}"
            code = code.replace(savefig_matches[0], output_sentence)

        return code, len(savefig_matches) > 0

    def sandbox(self, code, dataset_paths, question, output_dir, data_source, dataset_id, question_id, turn, agent_flows=None):
        # Error handling if needed
        output, error = self._execute(code)
        if error != "":
            print(f"ERROR::::::::::::::::::::::::::: {error}")
            code, output, flows = self.error_handling_agent(error, code)
            agent_flows["error_handling_agent"] = flows

        # Visualization handling if needed
        splits = str(output_dir).split("/")
        output_image_dir = "/".join(splits) + f"/{data_source}/{dataset_id}/rank{question_id}/turn{turn}/"
        output_image_paths = []
        if Path(output_image_dir).exists():
            output_image_paths = list(Path(output_image_dir).glob("*.png"))
            output_image_paths = [str(p) for p in output_image_paths]

        for output_image_path in output_image_paths:
            code, flows = self.visualization_agent(output_image_path, code, question)
            agent_flows["visualization_agent"] = flows

        return code, output_image_paths, output, agent_flows
    
    def _save_answers(self, answers, codes, iteration, data_source, dataset_id, rank):
        Path(f"{self.output_dir}/{data_source}/{dataset_id}/rank{rank}/iter{iteration}").mkdir(exist_ok=True, parents=True)
        output_answer_path = f"{self.output_dir}/{data_source}/{dataset_id}/rank{rank}/iter{iteration}/answers.json"
        answers = {"answers": answers}
        json_save(output_answer_path, answers)

        for i, code in enumerate(codes):
            if code is None:
                code = ""
            if isinstance(code, list): print(code)
            with open(f"{self.output_dir}/{data_source}/{dataset_id}/rank{rank}/iter{iteration}/code_{i}.py", 'w') as f:
                f.write(code)

    def _save_agent_flows(self, agent_flows, data_source, dataset_id, rank):
        output_answer_path = f"{self.output_dir}/{data_source}/{dataset_id}/rank{rank}/agent_flows.json"
        json_save(output_answer_path, agent_flows)
    
    def code_agent_only(self, dfs, dataset_paths, question, metadata, data_source, dataset_id, question_id, external_knowledge=None):
        if isinstance(question, list):
            questions = question
        else:
            questions = question.split("|")
        
        qa_pairs = []
        answers = []
        codes = []

        for turn, question in enumerate(questions):
            if question in ["", " "]: continue

            code = self.code_agent.evaluate(dfs, metadata, data_source, question, external_knowledge, qa_pairs)
            if code is None: return qa_pairs, codes
            code, chart_exist = self.fill_code(code, dataset_paths, self.output_dir, data_source, dataset_id, question_id, turn=turn)
            output, error = self._execute(code)

            splits = str(self.output_dir).split("/")
            output_image_dir = "/".join(splits) + f"/{data_source}/{dataset_id}/rank{question_id}/turn{turn}/"
            output_image_paths = []
            if Path(output_image_dir).exists():
                output_image_paths = list(Path(output_image_dir).glob("*.png"))
                output_image_paths = [str(p) for p in output_image_paths]

            qa_pairs.append(question)
            codes.append(code)

            if len(output_image_paths) > 0:
                qa_pairs.append(output_image_paths)
                answers.append(output_image_paths)
            else:
                qa_pairs.append(output.lstrip("\n").rstrip("\n"))
                answers.append(output.lstrip("\n").rstrip("\n"))

        return qa_pairs, codes

    def __call__(self, dfs, dataset_paths, question, metadata, data_source, dataset_id, question_id, external_knowledge=None, iteration=0):
        if isinstance(question, list):
            questions = question
        else:
            questions = question.split("|")

        agent_flows = []
        qa_pairs = []
        codes = []
        answers = []

        for turn, question in enumerate(questions):
            print(question)
            agent_flow = {}
            if question in ["", " "]: continue

            dfs_copy = [df.copy() for df in dfs]

            code = self.code_agent.evaluate(dfs_copy, metadata, data_source, question, external_knowledge, qa_pairs)
            code, chart_exist = self.fill_code(code, dataset_paths, self.output_dir, data_source, dataset_id, question_id, turn=self.turn)
            agent_flow["coding_agent"] = {"codes": [code]}

            #TODO: More convinced way to check whether the question asks insights or not
            code, output_image_paths, output, agent_flow = self.sandbox(code, dataset_paths, question, self.output_dir, data_source, dataset_id, question_id, self.turn, agent_flows=agent_flow)

            if len(output_image_paths) == 0:
                # Table reflection agent for non-visualization tasks
                judge, code = self.reflection_agent.evaluate(code, output, dfs_copy, metadata, data_source, question, external_knowledge, qa_pairs)
                if not judge:
                    output, _ = self._execute(code)
                    agent_flow["reflection_agent"] = {"code": [code], "output": [output]}

            qa_pairs.append(question)
            if len(output_image_paths) > 0:
                qa_pairs.append(output_image_paths)
                answers.append(output_image_paths)
            else:
                output_format = question.split("Format: ")
                if len(output_format) > 1:
                    output_format = output_format[1].rstrip("\n")
                    if output_format in output:
                        output = output.split(output_format)[1]
                qa_pairs.append(output.lstrip("\n").rstrip("\n"))
                answers.append(output.lstrip("\n").rstrip("\n"))

            codes.append(code)
            agent_flows.append(agent_flow)

            self.turn += 1

        self._save_answers(answers, codes, iteration, data_source, dataset_id, question_id)
        self._save_agent_flows(agent_flows, data_source, dataset_id, question_id)

        return qa_pairs, codes
