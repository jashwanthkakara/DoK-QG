from GEval import GEval
from Agent import *
from db import *
from prompts import *
import re,json,ast, random
def extract_dict_string(model_output):
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
            return 'anything'
    if('hobby' not in probs):
        return 'anything'
    return probs['hobby']

def parse_ai_output(raw_text):
    if not raw_text:
        print("[DEBUG] parse_ai_output: raw_text is empty")
        return []

    print(f"[DEBUG] parse_ai_output: Input length: {len(raw_text)} characters")
    print(f"[DEBUG] parse_ai_output: First 500 chars: {raw_text[:500]}")

    # More lenient patterns - try multiple variations
    # Pattern 1: Standard <Question>...</Question> format
    q_pattern1 = r'<(?:[Qq]uestion)>(.*?)(?:</[Qq]uestion>|(?=<[Qq]uestion>|<[Aa]nswer>|$))'
    a_pattern1 = r'<(?:[Aa]nswer)>(.*?)(?:</[Aa]nswer>|(?=<[Qq]uestion>|<[Aa]nswer>|$))'
    
    # Pattern 2: Without closing tags
    q_pattern2 = r'<(?:[Qq]uestion)>(.*?)(?=<[Qq]uestion>|<[Aa]nswer>|$)'
    a_pattern2 = r'<(?:[Aa]nswer)>(.*?)(?=<[Qq]uestion>|<[Aa]nswer>|$)'

    questions = re.findall(q_pattern1, raw_text, re.DOTALL)
    answers = re.findall(a_pattern1, raw_text, re.DOTALL)
    
    # If no matches, try pattern 2
    if not questions:
        questions = re.findall(q_pattern2, raw_text, re.DOTALL)
        answers = re.findall(a_pattern2, raw_text, re.DOTALL)

    # Fallback: markdown-style ### QUESTION / ### ANSWER or "Answer:" / "ANSWER"
    if not questions:
        md_answer = re.search(
            r'(?i)(?:###\s*)?(?:ANSWER|Answer)\s*\n\s*(.*)',
            raw_text,
            re.DOTALL
        )
        md_question = re.search(
            r'(?i)(?:###\s*)?(?:QUESTION|Question)\s*\n\s*(.*?)(?=(?:###\s*)?(?:ANSWER|Answer)\s*\n|\Z)',
            raw_text,
            re.DOTALL
        )
        if md_answer:
            a_text = md_answer.group(1).strip()[:2000]
            if md_question:
                q_text = md_question.group(1).strip()
            else:
                before = raw_text[:md_answer.start()].strip()
                q_text = before if len(before) < 2000 else before[:1997] + "..."
            if not q_text:
                q_text = "Question (format not parsed; see synthesis output)"
            questions = [q_text]
            answers = [a_text] if a_text else ["No answer provided."]
        elif re.search(r'(?i)(?:###\s*)?(?:ANSWER|Answer)\s*', raw_text):
            lines = raw_text.strip().split('\n')
            for i, line in enumerate(lines):
                if re.match(r'(?i)^(?:###\s*)?(?:ANSWER|Answer)\s*$', line.strip()) and i + 1 < len(lines):
                    questions = ["Question (see synthesis output)"]
                    answers = ['\n'.join(lines[i + 1:]).strip()[:2000]]
                    break
    
    print(f"[DEBUG] parse_ai_output: Found {len(questions)} questions, {len(answers)} answers")

    results = []
    
    # We loop based on the number of questions found
    for i in range(len(questions)):
        q_raw = questions[i]
        a_raw = answers[i] if i < len(answers) else "No answer provided."

        # CLEANUP: Remove any stray closing tags the AI might have actually included
        q_clean = re.sub(r'</?[Qq]uestion/?>', '', q_raw).strip()
        a_clean = re.sub(r'</?[Aa]nswer/?>', '', a_raw).strip()

        # CLEANUP: Remove AI artifacts like "**Question 1**" or "Note:"
        q_clean = re.sub(r'(?i)(\*\*Question\s*\d+\*\*|Question\s*\d+:|###.*?\n)', '', q_clean).strip()
        a_clean = re.sub(r'(?i)(\*\*Answer\*\*|Answer:|Note:.*$)', '', a_clean).strip()

        if q_clean:  # Only add if question is not empty
            results.append({
                "question": q_clean,
                "answer": a_clean
            })

    print(f"[DEBUG] parse_ai_output: Returning {len(results)} parsed results")
    return results

LLAMA_70B='ollama@llama3.3:70b@10.129.7.47:11434'
LLAMA_70B='ollama@llama3.2:3b@localhost:11434'

dok_alignment = GEval(
    model=LLAMA_70B,
    likert_scale=[1, 2, 3, 4, 5]  # or [1..7]
)

theme_alignment = GEval(
    model=LLAMA_70B,
    likert_scale=[1, 2, 3, 4, 5]  # or [1..7]
)

appropriateness = GEval(
    model=LLAMA_70B,  
    likert_scale=[1, 2]  # or [1..7]
)

correctness_deepseek = GEval(
    model=LLAMA_70B,# 'ollama@deepseek-r1:8b@10.129.7.47:11434',
    likert_scale=[1, 2]  # or [1..7]
)

correctness_olmo = GEval(
    model=LLAMA_70B,# 'ollama@olmo-3.1:32b@10.129.7.47:11434',
    likert_scale=[1, 2]  # or [1..7]
)

def get_alignment_score(req,q):
    theme_score = theme_alignment.evaluate(
        task_description=f"You are to determine whether the given question and answer pair is a standard NCERT 10th, 11th or 12th standard question or not along whether it is aligned to the given theme or not.\n Theme :{req['theme']}",
        evaluation_parameter="You to rate how well it is aligned on a scale of 1 to 5. A score of 1 indicates low alignemtn while a score of 5 indicates high alignment.",
        question=q['question'],
        answer=q['answer']
    )
    
    dok_score = dok_alignment.evaluate(
        task_description=f'''You are to evaluate the DoK level alignment of a question. 
        You must adhere to the following definitions for the requested DEPTH:
        - DOK 1 (Recall/Remember): Recall of a fact, term, or property. (e.g., Define, List, State)
        - DOK 2 (Skills & Concepts/Understand & Apply): Use of information or conceptual knowledge. (e.g., Describe, Classify, Solve routine problems)
        - DOK 3 (Strategic Thinking/Analyze & Evaluate): Reasoning, planning, and using evidence. (e.g., Explain why, Non-routine problem solving, Compare/Contrast phenomena)
        - DOK 4 (Extended Thinking/Create): Complex synthesis and connection across chapters. (e.g., Create a model, Design an experiment, Critique a theoretical framework)

The provided bloom level is {req['dok']}.''',
        evaluation_parameter="You to rate how well it is aligned on a scale of 1 to 5. A score of 1 indicates low alignemtn while a score of 5 indicates high alignment.",
        question=q['question'],
        answer=q['answer']
    )

    appropriateness_score = appropriateness.evaluate(
        task_description=GUARDRAILS_PROMPT,
        evaluation_parameter="You to rate whether the question is appropriate or not on a scale of 1 to 2. A score of 1 indicates inappropriateness while a score of 2 indicates appropriate question.",
        question=q['question'],
        answer=q['answer']
    )

    if(req['subject'] in ['Math','Physics']):
        correctness_score = correctness_deepseek.evaluate(
            task_description=f"You are to determine whether the given question and answer pair is valid or not. Try to solve the question without looking at the answer and then verify with the given answer.",
            evaluation_parameter="You have to rate its correctness level on a scale of 1 to 2. A score of 1 indicates incorrect question while a score of 2 indicates correct question.",
            question=q['question'],
            answer=q['answer']
        )
    else:
        correctness_score = correctness_olmo.evaluate(
            task_description=f"You are to determine whether the given question and answer pair is valid or not. Try to solve the question without looking at the answer and then verify with the given answer.",
            evaluation_parameter="You have to rate its correctness level on a scale of 1 to 2. A score of 1 indicates incorrect question while a score of 2 indicates correct question.",
            question=q['question'],
            answer=q['answer']
        )
    output = {
        'dok':dok_score,
        'theme':theme_score,
        'appropriateness': appropriateness_score,
        'correctness': correctness_score
        }
    return output
        
QID=1
candidate_models=[LLAMA_70B]
for candidate_model in candidate_models:
    qg_agent = Agent(model=candidate_model)
    for standard in chapter_map.keys():
        for subject in chapter_map[standard].keys():
            student_agent = FixedAgent(model=LLAMA_70B,prompt=STUDENT_HOBBY_PROMPT.format(hobbies_list=hobbies,standard=standard,subject=subject))
            for dok in [1,2,3,4]:
                for idx in range(500):
                    hobby = student_agent._fixed_call()
                    hobby = extract_dict_string(hobby)
                    chapter = random.choice(chapter_map[standard][subject])
                    qType = random.choice(question_types)
                    prompt=QG_prompt.format(
                        subject=subject,
                        chapter=chapter,
                        qType=qType,
                        lang_block='English',
                        theme=hobby,
                        depth=dok
                    )

                    questions = qg_agent._call_model(prompt)
                    questions = parse_ai_output(questions)

                    for question in questions:
                        metadata={
                            'subject':subject,
                            'chapter':chapter,
                            'qType':qType,
                            'lang_block':'English',
                            'theme':hobby,
                            'dok':dok
                        }
                        scores = get_alignment_score(metadata,question)
                        insert_question(QID, question['question'], question['answer'], candidate_model, metadata=metadata, scores=scores)                        
                        QID = QID+1
                        print(QID, question['question'], question['answer'], candidate_model, metadata,scores)                        
                        exit(0)
            del student_agent
    del qg_agent
