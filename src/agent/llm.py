import torch
import os
import base64
import time
import re
import json
import requests
import pandas as pd
import ast
import ollama

from PIL import Image

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import pipeline as Pipe
from qwen_vl_utils import process_vision_info
from ollama import chat

from mistral_common.protocol.instruct.messages import (
    SystemMessage, UserMessage
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from huggingface_hub import hf_hub_download


class LLM:
    def __init__(self, method="llama", multimodal=False, system_prompt_enable=False):
        self.method = method
        self.ollama = False
        self.gptoss = False
        self.mistral = False
        if "/" in method:
            if multimodal:
                self.model, self.text_tokenizer = self._qwen_vl_initialize(method)
            elif method.split("/")[0] == "ollama":
                self.model = method.split("/")[1]
                ollama.pull(self.model)
                self.ollama = True
            elif method.split("/")[0] == "mistralai":
                self.mistral = True
                self.model, self.tokenizer, self.system_prompt = self._mistral_initialize(method)
            elif method.split("/")[0] == "openai":
                self.gptoss = True
                self.pipeline = self._gpt_oss_initialize()
            else:
                self.model, self.tokenizer = self._llm_initialize(method)
            self.multimodal = multimodal
        else:
            self.azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            self.gpt4_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            self.gpt4_mini_endpoint = os.environ.get("AZURE_OPENAI_MINI_ENDPOINT")
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
            self.gemini_endpoint = os.environ.get("GEMINI_ENDPOINT")
            self.gemini_pro_endpoint = os.environ.get("GEMINI_PRO_ENDPOINT")

        self.system_prompt_enable = system_prompt_enable
        self.output_format_conversion = {
            "@numerical_value": "Numerical value without rounding",
            "@text": "Simple words or sentence without redundant texts",
            "@list()[a, b, c, ...]": "List values specified as a, b, c, ...",
            "@list(ascending, descending, alphabetic, or mentioned order)": "List values or elements in ascending or descending",
            "@list_tuples(ascending, descending, alphabetic, or mentioned order)[k, v]": "List tuples of k and v in ascending, descending, or alphabetic order",
            "@dictionary(ascending, descending, alphabetic, or mentioned order)[k, v]": "Dictionary where key is k, and value is v",
            "@line_plot(ascending, descending, alphabetic, or mentioned order)[x, y]#T[a, b, ...]": "Line plot where x-axis is x and y-axis is y, legends are a, b, ..., and T is legend title",
            "@scatter_plot(ascending, descending, alphabetic, or mentioned order)[x, y]#T[a, b, ...]": "Scatter plot where x-axis is x and y-axis is y, legends are a, b, ..., and T is legend title",
            "@heatmap": "Draw heatmap",
            "@heatmap[x, y]": "Draw heatmap where x-axis us x and y-axis is y",
            "@bar_chart(ascending, descending, alphabetic, or mentioned order)[x, y]#T[a, b, ...]": "Bar chart where x-axis is x and y-axis is y, legends are a, b, ..., and T is legend title",
            "@stacked_bar_chart(ascending, descending, alphabetic, or mentioned order)[x, y]#T[a, b, ...]": "Stacked bar chart where x-axis is x and y-axis is y, legends are a, b, ..., and T is legend title",
            "@pi_chart": "Draw pi chart",
            "@box_plot[x, y]": "Draw box plot where x-axis is x and y-axis is y",
            "@double_axis_chart[X, Y]": "If the visualization is double-axis, please list two chart format based on the above.",
            "@list_subplots[X, Y, Z, ...]": "Draw the single graph including subplots composing of X, Y, Z...",
            "@list_graphs[X, Y, Z, ...]": "Draw multiple graphs (X, Y, Z) as separated graph files"
        }

        if self.mistral:
            self.system_prompt += "\n\nGiven question includes specific output formats starting with @ as below, so please follow the format. **IMPORTANT: The final answer MUST NOT include the output format tags (e.g., @numerical_value, @list_tuples(...)[...]) but only the value itself**. 'mentioned order' means that the order should follow one mentioned in the question or answer of the previous question (if the question is multi-turn). If you encounter the numerical values, please do not do trimming or rounding.\n"
        else:
            self.system_prompt = "Given question includes specific output formats starting with @ as below, so please follow the format. **IMPORTANT: The final answer MUST NOT include the output format tags (e.g., @numerical_value, @list_tuples(...)[...]) but only the value itself**. 'mentioned order' means that the order should follow one mentioned in the question or answer of the previous question (if the question is multi-turn). If you encounter the numerical values, please do not do trimming or rounding.\n"
        for k, v in self.output_format_conversion.items():
            self.system_prompt += f"{k}: {v}"

    @staticmethod
    def _wrap_json(response):
        try:
            if "```json" in response:
                response = response.lstrip("```json\n").rstrip("\n```")
            response = response.replace("\n", "").replace("    ", "")
            if '"[' in response:
                response = response.replace('"[', '[').replace(']"', ']')
            response = ast.literal_eval(response)
            return response
        except Exception as e:
            print(f"JSON Convert Error: {e}")
            return response

    @staticmethod
    def _wrap_code(response):
        try:
            if response.startswith("```python"):
                response = response.lstrip("```python").rstrip("```")
            elif "```python" in response:
                pattern = r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'
                result = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
                response = result[0]
                response = response.lstrip("```python").rstrip("```")
            elif response.startswith("<think>"):
                pattern = r"<answer>(.*?)</answer>"
                matches = re.findall(pattern, response)
                response = matches[0]
            return response
        except Exception as e:
            return response

    def _ovis_initialize(self, model):
        model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.bfloat16,
                                                    #  load_in_8bit=True,
                                                    #  low_cpu_mem_usage=True,
                                                     device_map="auto",
                                                     multimodal_max_length=32768,
                                                     trust_remote_code=True)
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        return model, text_tokenizer, visual_tokenizer
    
    def _qwen_vl_initialize(self, model_path):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )

        processor = AutoProcessor.from_pretrained(model_path, min_pixels=1280*28*28, max_pixels=16384*28*28)

        return model, processor
    
    def _llm_initialize(self, model):
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto",
                                                     trust_remote_code=True
                                                     )

        return model, tokenizer

    def _gpt_oss_initialize(self):
        model_id = "openai/gpt-oss-20b"

        pipe = Pipe(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )

        return pipe
    
    def _mistral_initialize(self, model):
        file_path = hf_hub_download(repo_id=model, filename="SYSTEM_PROMPT.txt")
        with open(file_path, "r") as file:
            system_prompt = file.read()

        tokenizer = MistralTokenizer.from_hf_hub(model)
        model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto",
                                                     trust_remote_code=True)
        
        return model, tokenizer, system_prompt
    
    def _execute(self, data, index, api_key, endpoint):
        headers = {
                'Content-type': 'application/json',
                'api-key': api_key,
        }

        generated_text = None
        limit_count = 0

        while generated_text is None and limit_count < 8:
            response = requests.post(endpoint,
                                     headers=headers,
                                     json=data).json()

            if response.get("choices") is not None:
                generated_text = response["choices"][0]["message"]["content"]
            elif response.get("candidates"):
                generated_text = response["candidates"][0]["content"]["parts"][0]["text"]
            else:
                time.sleep(60)
                limit_count += 1

        response_dict = {
            "generated_text": generated_text,
            "index": index
        }

        return response_dict

    def _generate_ovis(self, prompt, image_paths=None):
        if image_paths is not None:
            images = [Image.open(image_path) for image_path in image_paths]
            query = f'<image>\n{prompt}'
        else:
            images = None
            query = f'{prompt}'
        max_partition = 9

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
        pixel_values = [pixel_values]

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=0.0,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)

        return output

    def _generate_qwen_vl(self, prompt, image_paths):
        system_prompt = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since 1+1=2, so the answer is 2. </think><answer> 2 </answer>, which means assistant's output should start with <think> and end with </answer>."

        system_prompt += f"\n\n{self.system_prompt}"

        generate_kwargs = dict(
            max_new_tokens=2048,
            top_p=0.001,
            top_k=1,
            temperature=0.01,
            repetition_penalty=1.0
        )

        # Prepare input with image and text
        messages = [{"role": "system", "content": system_prompt}]
        data = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                data.append({"type": "image", "image": image_path})
        message = {"role": "user", "content": data}
        messages.append(message)

        # Preparation for inference
        text = self.text_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.text_tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.text_tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

    def _generate_local_llm(self, prompt):
        if self.system_prompt_enable:
            messages = [
                {"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=4000, temperature=1e-9)
        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
    
    def _generate_ollama(self, prompt):
        if self.system_prompt_enable:
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = chat(model=self.model, messages=messages)
        response =  response['message']['content']

        if "</think>" in response: response = response.split("</think>")[1]

        return response
    
    def _generate_mistral(self, prompt):
        tokenized = self.tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[
                    SystemMessage(content=self.system_prompt),
                    UserMessage(content=prompt),
                ],
            )
        )

        output = self.model.generate(
            input_ids=torch.tensor([tokenized.tokens]),
            max_new_tokens=4000,
            temperature=1e-9
        )[0]

        decoded_output = self.tokenizer.decode(output[len(tokenized.tokens):])
        
        return decoded_output
    
    def _generate_gptoss(self, prompt):
        messages = [
            {"role": "user", "content": prompt},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=4000,
        )

        print(outputs[0]["generated_text"][-1])
        
        return outputs[0]["generated_text"][-1]

    def _generate_gemini(self, prompt, image_paths=None, endpoint=None):
        data = [{"text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    d = base64.b64encode(image_file.read()).decode("utf-8")
                data.append({"inlineData": {"mimeType": "image/jpeg", "data": f"{d}"}})
        message = {}
        if self.system_prompt_enable:
            data_system = {"parts": [{"text": self.system_prompt}]}
            message["system_instruction"] = data_system
        message["contents"] = [{"role": "user", "parts": data}]
        message["generation_config"] = {"temperature": 0.0, "max_output_tokens": 3000, "presence_penalty": 1.0}

        generated_text = []
        indices = []

        for _ in range(1):
            response = self._execute(message, 0, self.gemini_api_key, endpoint)
            generated_text.append(response["generated_text"])
        output = generated_text[0]

        return output
    
    def _generate_gpt4(self, prompt, image_paths=None, temperature=0.0, endpoint=None):
        data = [{"type": "text", "text": prompt}]
        if image_paths is not None:
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    d = base64.b64encode(image_file.read()).decode("utf-8")
                data.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}})
        if self.system_prompt_enable:
            message = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": data}]
        else:
            message = [{"role": "user", "content": data}]
        body = {
            "messages": message,
            "max_tokens": 4000,
            "presence_penalty": 1.0,
            "temperature": temperature
        }

        generated_text = []
        indices = []

        for _ in range(1):
            response = self._execute(body, 0, self.azure_api_key, endpoint)
            generated_text.append(response["generated_text"])
        output = generated_text[0]

        return output

    def __call__(self, prompt, image_paths=None, temperature=0.0):
        if "/" in self.method:
            if self.multimodal:
                generated_text = self._generate_qwen_vl(prompt, image_paths=image_paths)
            elif self.ollama:
                generated_text = self._generate_ollama(prompt)
            elif self.mistral:
                generated_text = self._generate_mistral(prompt)
            elif self.gptoss:
                generated_text = self._generate_gptoss(prompt)
            else:
                generated_text = self._generate_local_llm(prompt)
        elif self.method == "gpt4":
            generated_text = self._generate_gpt4(prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt4_endpoint)
        elif self.method == "gpt4-mini":
            generated_text = self._generate_gpt4(prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt4_mini_endpoint)
        elif self.method == "gemini":
            generated_text = self._generate_gemini(prompt, image_paths=image_paths, endpoint=self.gemini_endpoint)
        elif self.method == "gemini-pro":
            generated_text = self._generate_gemini(prompt, image_paths=image_paths, endpoint=self.gemini_pro_endpoint)

        return generated_text
    
    def generate_format(self, prompt, image_paths=None, temperature=0.0, format="code"):
        if "/" in self.method:
            if self.multimodal:
                generated_text = self._generate_qwen_vl(prompt, image_paths=image_paths)
            elif self.ollama:
                generated_text = self._generate_ollama(prompt)
            elif self.mistral:
                generated_text = self._generate_mistral(prompt)
            elif self.gptoss:
                generated_text = self._generate_gptoss(prompt)
            else:
                generated_text = self._generate_local_llm(prompt)
        elif self.method == "gpt4":
            generated_text = self._generate_gpt4(prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt4_endpoint)
        elif self.method == "gpt4-mini":
            generated_text = self._generate_gpt4(prompt, image_paths=image_paths, temperature=temperature, endpoint=self.gpt4_mini_endpoint)
        elif self.method == "gemini":
            generated_text = self._generate_gemini(prompt, image_paths=image_paths, endpoint=self.gemini_endpoint)
        elif self.method == "gemini-pro":
            generated_text = self._generate_gemini(prompt, image_paths=image_paths, endpoint=self.gemini_pro_endpoint)

        if generated_text is None: return generated_text
        if generated_text.startswith("-") or generated_text == "OK":
            return generated_text
        
        if format == "code":
            generated_text = self._wrap_code(generated_text)
        elif format == "json":
            generated_text = self._wrap_json(generated_text)

        return generated_text
    
    def ensemble(self, prompt, image_paths=None):
        results = {}
        results["gpt4"] = self._wrap_json(self._generate_gpt4(prompt, image_paths=image_paths, endpoint=self.gpt4_endpoint))
        results["gpt4-mini"] = self._wrap_json(self._generate_gpt4(prompt, image_paths=image_paths, endpoint=self.gpt4_mini_endpoint))
        results["gemini"] = self._wrap_json(self._generate_gemini(prompt, image_paths=image_paths, endpoint=self.gemini_endpoint))
        results["gemini-pro"] = self._wrap_json(self._generate_gemini(prompt, image_paths=image_paths, endpoint=self.gemini_pro_endpoint))

        return results
