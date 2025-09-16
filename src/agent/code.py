from src.agent.sequence import sequence_tables, sequence_external_knowledge, sequence_qa_pairs
from src.agent.llm import LLM


class CodeAgent:
    def __init__(self, model="gpt4", local_llm=False, head_only=False):
        self.llm = model
        self.local_llm = local_llm
        self.head_only = head_only

    def construct_prompt(self, dfs, metadata, data_source, question, external_knowledges=None, previous_qa_pairs=None):
        dataset_title = metadata["dataset_title"]
        dataset_description = metadata["dataset_description"]
        publisher = metadata["publisher"]

        table_information = sequence_tables(dfs, metadata, self.head_only)

        if external_knowledges is not None:
            knowledge = sequence_external_knowledge(external_knowledges)

        prompt = f"""You are a top data analysis for the tabular data. Given a tabular data and question related to the table, your task is to generate the Python code to answer the question with the use of methods in Pandas or produce just text-based insights or summary especially when you are required to generate insights. The question must be answered by using the given tabular data except for the insight answering.
- If the answer aims to generate the image files (e.g. chart, graph, or figure), please set the output file name as 'output_[index].png', where 'index' is the index of single or multiple files, and please do not include output file names except for in the savefig function. Also, please follow the tips below. Create a clear and easy-to-read graph for human when the task is visualization, and try to use legend as long as the number of labels is not large. When you call 'savefig' function, you must set the following parameter: "bbox_inches='tight'".
- If the answer aims to generate the insights or summary from the multi-turn conversation, please output only text (not Python code) in the bullet points starting hyphen that will be saved as text file. Output format must be "- (insight1)\n- (insight2)\n- ..." without redundant explanation.
- If the question includes the specified output format, please follow the format without adding redundant texts. Please include the answer in the print statement in the last line, and do not use the print statement except for the last line. Please do not use print statement if the answer aims to generate the image files.
- Please do not truncate the decimal point unless the question requires it.
- The output is only Python code without additional explanation.
- In the generated Python code, the input file path is 'data_[index].csv', where 'index' is the index of the table starting from 1 (1, 2, 3, 4...) depending on the number of available datasets, so for instance the code includes 'pd.read_csv('data_1.csv')' when loading the csv file. The tentative path will be replaced by the actual path afterwards, so please just follow the rule.
- Please include print statement for the final answer in the last line.
- Please do not include try-except statement.

Question: {question}

Data publisher: {publisher}
Data source: {dataset_title}
Data source description: {dataset_description}
"""
        
        if len(previous_qa_pairs) > 0:
            prompt += "\nThis question is a part of multi-turn conversation. Following is the previous question and answer pairs. You can refer to them to understand the contexts.\n"
            pairs, image_paths = sequence_qa_pairs(previous_qa_pairs)
            prompt += pairs
        else:
            image_paths = []
        
        for t_info in table_information:
            prompt += f"{t_info}\n\n"

        if external_knowledges is not None:
            prompt += "Following is the external knowledge on the tabular information. Please use the knowledge to make the ID-like or ambiguous words clear.\n"
            prompt += knowledge
            
        return prompt, image_paths
    
    def evaluate(self, dfs, metadata, data_source, question, external_knowledges=None, previous_qa_pairs=None):
        prompt, image_paths = self.construct_prompt(dfs, metadata, data_source, question, external_knowledges, previous_qa_pairs)

        if self.local_llm:
            code = self.llm.generate_format(prompt, format='code')
        else:
            code = self.llm.generate_format(prompt, image_paths, format='code')

        return code
