from src.agent.sequence import sequence_external_knowledge, sequence_qa_pairs, sequence_tables


class TableReflectionAgent:
    def __init__(self, model="gpt4", local_llm=False, head_only=False):
        self.llm = model
        self.local_llm = local_llm
        self.head_only = head_only
        
    def _construct_prompt(self, code, answer, dfs, metadata, data_source, question, external_knowledges=None, previous_qa_pairs=None):
        dataset_title = metadata["dataset_title"]
        dataset_description = metadata["dataset_description"]
        publisher = metadata["publisher"]

        table_information = sequence_tables(dfs, metadata, self.head_only)

        if external_knowledges is not None:
            knowledge = sequence_external_knowledge(external_knowledges)

        if len(previous_qa_pairs) > 0:
            qa_pairs = "\nThis question is a part of multi-turn conversation. Following is the previous question and answer pairs. You can refer to them to understand the contexts.\n"
            pairs, image_paths = sequence_qa_pairs(previous_qa_pairs)
            qa_pairs += pairs
        else:
            qa_pairs = "No previous pairs"
            image_paths = []

        prompt = f"""You are a top data analysis for the tabular data. Given the question asking about table data, the generated code to answer the question, and the answer, please revise the code toward the correct answer if the answer would be wrong. Table information is also given to contextualize. If the given answer is correct and the code does not need to be revised, please answer only 'OK!' without additional text. If the code needs to be revised, please answer only Python code without additional explanation and any markdown formatting. Following perspectives are included in the correct answer or codes.
- Given questions include particular output formats. Following the output format is imperative without adding redundant texts.
- Answers are included in the print statement.
- Please do not include try-except statement.
- The question is a part of the multi-turn questions, the generated code should include context produced from the previous QAs if needed.
- Columns in the table data would include the multiple hierarchical categories (e.g. total, male, female in gender column). Please be careful about the aggregation of the column if needed.
- Columns in the table data would represent numerical values as string values (e.g. comma is inserted). Please convert them into numerical values appropriately if needed.
- Columns in the table data would include special characters as a replacement of NaN values (e.g. x, -, -99999, etc). Please replace them with NaN values if needed.
- If the answer is NaN or None, the filtering conditions might be wrong.
- If the output format is just numerical values, they should not be truncated or rounded.

## QA pair for the question
Target question: {question}
Generated code that would be revised:\n{code}
Answer: {answer}

## QA pairs so far in a multi-turn conversation
{qa_pairs}

## Table information
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

    def evaluate(self, code, answer, dfs, metadata, data_source, question, external_knowledges=None, previous_qa_pairs=None):
        prompt, image_paths = self._construct_prompt(code, answer, dfs, metadata, data_source, question, external_knowledges, previous_qa_pairs)
        if self.local_llm:
            response = self.llm.generate_format(prompt, format='code')
        else:
            response = self.llm.generate_format(prompt, image_paths, format='code')
        count = 0

        if response.lstrip("\n").rstrip("\n") == "OK!": return True, code
        elif "OK!" in response: return True, code

        if not response.lstrip("\n").startswith("import"): return True, code

        if response == "OK":
            return True, response
        else:
            return False, response