import json,random
from typing import Dict, List
from groq import Groq
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch,re
from prompts import *
import requests,os
groq_id_model_mapping = {
    "groq-llama-8b": "llama-3.1-8b-instant",
    "groq-llama-70b": "llama-3.3-70b-versatile",
    "rag-piped-groq-70b": "llama-3.3-70b-versatile",
    "groq-llama-guard": "meta-llama/llama-guard-4-12b",

    # Groq – GPT OSS
    "groq-gpt-oss-120b": "openai/gpt-oss-120b",
    "groq-gpt-oss-20b": "openai/gpt-oss-20b",
}
class Agent:
    def __init__(
        self,
        model: str,
        groq_api_key: str='',
    ):
        """
        Parameters
        ----------
        groq_api_key : str
            Groq API key
        model : str
            Groq model name
        """
        print('Processing : ', model,model.startswith('ollama'))
        if(model.startswith('ollama')):
            self.model=model
            self._call_model=self._call_ollama
        else:
            is_url = lambda s: all([urlparse(s).scheme, urlparse(s).netloc])
            self.client = Groq(api_key=groq_api_key)
            self.model_name=model
            if(is_url(model)):
                self._call_model=self._call_vllm
                self.model = model
            elif('groq' in model):
                self._call_model=self._call_groq
                self.model = groq_id_model_mapping[model]
            else:
                self._call_model=self._call_hf
                self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model,
                    trust_remote_code=True,
                    device_map="auto"
                )
        
    # --------------------------------------------------
    # Internal Groq streaming call (PATCHED)
    # --------------------------------------------------
    def _call_model(self,prompt: str) -> str:
        pass 

    def _call_groq(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )

        model_output = ""

        for chunk in completion:
            if (chunk.choices[0].delta.content):
                model_output += chunk.choices[0].delta.content
        return model_output.strip()

    def _call_vllm(self, prompt: str) -> str:
        data = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                }

        resp = requests.post(self.model, json=data)
        resp=resp.json()
        resp=resp['choices'][0]['message']['content']
        remove_think = lambda s: re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
        resp=remove_think(resp)
        return resp.strip()

    def _call_ollama(self, prompt: str) -> str:
        _, model_id, ip = self.model.split('@')
        url = f"http://{ip}/api/generate"
        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": False
        }
        print(model_id,url)
        response = requests.post(url, json=payload)
        response = response.json()["response"].strip()
        # os.system(f"ollama stop {model_id}")
        return response
        
    def _call_hf(self, prompt: str) -> str:
        conversation = [
            {
                "content": "You are an expert who knows about NCERT syllabus of 10th standard for physics.",
                "role": "system"
            },
            {
                "content": prompt,
                "role": "user"
            }
        ]

        # padding special token
        inputs = self.tokenizer.apply_chat_template(
            conversation=conversation,
            return_tensors="pt",
            add_generation_prompt=True 
        )
        inputs = inputs.to(self.model.device)

        # --- Generate output ---
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.6,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False
            )

        # Get only the generated tokens (exclude the prompt length)
        generated_tokens = output[0][inputs.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text.strip()

class FixedAgent(Agent):
    def __init__(
        self,
        model: str,
        prompt: str='',
        groq_api_key: str='',
        
    ):
        super().__init__(model=model,groq_api_key=groq_api_key)
        self.prompt=prompt
    def _fixed_call(self):
        print(self.model)
        return self._call_model(self.prompt)

