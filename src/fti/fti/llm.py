import concurrent.futures as cf
import json
import os

import requests
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")


class LargeLanguageModel:
    def __init__(self,
                 model="gpt4"
                 #model="HuggingFaceH4/zephyr-7b-alpha"
                 ):

        self.model = model
        if not model == "gpt4":
            self.pipeline, self.eos_token_id = self._initialize_llm(model)

    @staticmethod
    def _initialize_llm(model_name):
        model_4bit = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            #load_in_8bit=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_4bit,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        return pipeline, tokenizer.eos_token_id

    @staticmethod
    def _execute(data, index):
        headers = {
                'Content-type': 'application/json',
                'api-key': API_KEY,
        }
        response = requests.post(ENDPOINT,
                                 headers=headers,
                                 data=json.dumps(data)).json()
        print(response)
        generated_text = response["choices"][0]["message"]["content"]

        response_dict = {
            "generated_text": generated_text,
            "index": index
        }

        return response_dict

    def __call__(self, prompts):
        num_inputs = len(prompts)
        if num_inputs > 1:
            data = []
            for prompt in prompts:
                d = {"messages": [{"role": "system",
                                   "content": prompt}],
                     "max_tokens": 512,
                     "presence_penalty": 1.0,
                     "temperature": 1e-13,
                     "top_p": 1e-13,
                     "seed": 1234}
                data.append(d)

            generated_text = []
            indices = []

            with cf.ThreadPoolExecutor(max_workers=10) as executor:
                processes = {executor.submit(self._execute, query, i) for i, query in enumerate(data)}
                for result in cf.as_completed(processes):
                    result = result.result()
                    generated_text.append(result["generated_text"])
                    indices.append(result["index"])

            ordered_generated_text = []
            for i in range(len(prompts)):
                c_index = indices.index(i)
                ordered_generated_text.append(generated_text[c_index])

            generated_text = ordered_generated_text

        else:
            data = []
            for prompt in prompts:
                d = {"messages": [{"role": "system",
                                   "content": prompt}],
                     "max_tokens": 512,
                     "presence_penalty": 1.0,
                     "temperature": 0.0}
                data.append(d)

            generated_text = []
            indices = []

            for i, inputs in enumerate(data):
                response = self._execute(inputs, i)
                generated_text.append(response["generated_text"])

        return generated_text
