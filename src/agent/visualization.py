from src.agent.llm import LLM
from src.agent.utils import execute_code


class VisualizationAgent:
    def __init__(self, model="gpt4"):
        self.llm = model

    def evaluate(self, vis_path, code, question):
        prompt = f"""You are a top data analyst with much experience on Python and matplotlib. Given one generated figure that aims to answer the question '{question}', your task is to analyze the figures and judge whether the figures are aligning with the question and human understandable. If it is not, please revise the Python code to generate figure. Please do not include try-except statement. If it is OK, please answer only "OK!" without additional text.
Here are the perspectives to consider to judge whether the figure is human understandable or not:
- The figure should be clear without too many data points.
- There should not be overlapped colors in the figure.
- Axis labels should be clear and not too long.
- If the figure is a line chart, the line goes to the right direction without going back and forth.
- If the figure has small number of data points, the figure should be the scatter plot.

Here is the code to generate the figure. If you output the revised code, please only output the code without any other text and markdown formatting:
{code}

"""
        
        image_paths = [vis_path]
        generated_text = self.llm.generate_format(prompt, image_paths=image_paths, format='code')
        if generated_text.lstrip("\n").rstrip("\n") == "OK!":
            return True, code
        elif "OK!" in generated_text:
            return True, code
        else:
            return False, generated_text
        
    def __call__(self, vis_path, code, question):
        flows = {}
        judge_result, code = self.evaluate(vis_path, code, question)
        flows["codes"] = [code]
        flows["judge"] = [judge_result]
        fail_count = 0

        while not judge_result:
            _, _ = execute_code(code)
            judge_result, code = self.evaluate(vis_path, code, question)
            flows["codes"].append(code)
            flows["judge"].append(judge_result)
            fail_count += 1

            if fail_count == 3: break

        return code, flows
    

class VisualizationEvaluateAgent:
    def __init__(self, model="gpt4"):
        self.llm = LLM(method=model)

    def _aggregate(self, results):
        sum_score = 0
        for model, judge in results.items():
            if isinstance(judge, dict):
                if judge['judge'] == 'Same': sum_score += 1

        if sum_score >= 3: return True
        else: return False

    def evaluate(self, vis_path, target_path, pred_code, target_code):
        if isinstance(vis_path, str): return False
        if isinstance(target_path, str): return False

        output_format = {"judge": "Same or Different", "reason": "Reason for judgement"}
        output_format = str(output_format)

        num_vis = len(vis_path)
        num_tgt = len(target_path)

        prompt = f"""You are a top data analyst. Given the generated graph(s) (first {num_vis} images), the target graph(s) (last {num_tgt} images), and Python codes to generate them, your task is to judge whether the generated graph is the ROUGHLY same as the target graph based on the following perspectives attached with reasons of judgement. Output format is {output_format} in JSON format enclosed with double quotation.
- Chart type (e.g. Line chart, bar chart, pi chart...)
- Roughly similar axis name / legends (They should not be exactly the same, but rough similarity is preferred)
- Values of the characteristic data points should be matched
- IMPORTANT: Prioritize the trajectory of the graph because the graphs tend to be matched if the shapes of the graphs are the same
- Ignore the color scheme and the size of the chart
- Ignore the position (vertical or horizontal) of the subplots in the graph
- Ignore the marker in the line chart
- Ignore the grid of the chart
- Ignore the title of the graph
- Ignore the rotation of the axis 
- If the x-axis is year or date, as long as the minimum and maximum date is the same, it is regarded as the same in terms of the x-axis
- There is a case where tick marks are not explicitly drawn though the range of the axis is the same. In that case, please treat them as the same graph

Please answer "Same" or "Different". If the generated graph matched the target graph based on the above perspectives, please say "Same". Otherwise, please answer "Different". Please do not add explanation. Following is the Python codes for generated graph and target graph.
### Code for generated graph
{pred_code}

### Code for target graph
{target_code}
"""

        image_paths = vis_path + target_path
        response = self.llm.ensemble(prompt, image_paths)
        
        return self._aggregate(response)

