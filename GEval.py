import json,random
from typing import Dict, List
from groq import Groq
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch,requests,re
from Agent import Agent, groq_id_model_mapping
class GEval(Agent):
    def __init__(
        self,
        model: str,
        groq_api_key: str='',
        likert_scale: List[int] = None,
    ):
        """
        Parameters
        ----------
        groq_api_key : str
            Groq API key
        model : str
            Groq model name
        likert_scale : list[int]
            Likert scale values (default: [1,2,3,4,5])
        """
        super().__init__(model=model,groq_api_key=groq_api_key)
        self.likert_scale = likert_scale or [1, 2, 3, 4, 5]
        self.output_format_scale=self.generate_likert_probability_string()
        
    def generate_likert_probability_string(self):
        # Step 1: generate random positive numbers
        raw = [random.random() for _ in self.likert_scale]

        # Step 2: normalize so they sum to 1
        total = sum(raw)
        probs = [round(x / total, 2) for x in raw]

        # Step 3: adjust rounding drift to ensure exact sum = 1.00
        diff = round(1.0 - sum(probs), 2)
        probs[-1] = round(probs[-1] + diff, 2)

        # Step 4: build dict with string keys
        prob_dict = {
            str(k): v for k, v in zip(self.likert_scale, probs)
        }

        # Step 5: return pretty JSON string
        return json.dumps(prob_dict, indent=2)
    # --------------------------------------------------
    # 1. Generate Chain-of-Thought (CoT)
    # --------------------------------------------------
    def generate_cot(self, task_description: str, evaluation_parameter: str) -> str:
        prompt = f"""
You are an expert evaluator. 

Task:
{task_description}

Evaluation Parameter:
{evaluation_parameter}

Generate a concise chain-of-thought explaining how this task should be evaluated using the parameter.
"""

        return self._call_model(prompt)

    # --------------------------------------------------
    # 2. Generate Likert probabilities
    # --------------------------------------------------
    def generate_likert_probabilities(
        self,
        task_description: str,
        evaluation_parameter: str,
        question: str,
        answer: str,
        cot: str
    ) -> Dict[int, float]:

        scale_str = ", ".join(map(str, self.likert_scale))
        prompt = f"""
You are an expert evaluator.

Task:
{task_description}

Evaluation Parameter:
{evaluation_parameter}

Chain-of-Thought:
{cot}

Question:
{question}

Answer:
{answer}

Return ONLY valid JSON.

Return a probability distribution over the Likert scale [{scale_str}]
where probabilities sum to 1.

Format exactly like this:
<OUTPUT>
{self.output_format_scale}
</OUTPUT>
"""

        model_output = self._call_model(prompt)
        try:
            match = re.search(r"<OUTPUT>\s*(.*?)\s*</OUTPUT>", model_output, re.DOTALL)
            if match:
                extracted = match.group(1)
                model_output=extracted
        except:
            model_output=model_output.replace("<OUTPUT>",'').replace('</OUTPUT>','')
            extract = lambda s: {k: float(v) for k, v in re.findall(r'"(\d+)"\s*:\s*([0-9]*\.?[0-9]+)', s)}
            model_output = extract(model_output)

        try:
            probs = json.loads(model_output)
        except:
            try:
                probs = ast.literal_eval(model_output)
            except:
                print("FAILED PROBS : ",model_output)
                return {1:1.0}
        return {int(k): float(v) for k, v in probs.items()}

    # --------------------------------------------------
    # 3. Compute weighted average score
    # --------------------------------------------------
    def compute_weighted_score(self, probabilities: Dict[int, float]) -> float:
        return sum(
            score * probabilities.get(score, 0.0)
            for score in self.likert_scale
        )

    # --------------------------------------------------
    # 4. Full evaluation pipeline
    # --------------------------------------------------
    def evaluate(
        self,
        task_description: str,
        evaluation_parameter: str,
        question: str,
        answer: str
    ) -> float:
        if('param' in self.model.lower()):
            cot =''
        else:
            cot = self.generate_cot(task_description, evaluation_parameter)
        probabilities = self.generate_likert_probabilities(
            task_description,
            evaluation_parameter,
            question,
            answer,
            cot
        )
        return self.compute_weighted_score(probabilities)

