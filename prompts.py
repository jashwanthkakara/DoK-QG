QG_prompt = '''
    ### ROLE\n
    Act as an expert Academic Assessment Designer specializing in NCERT/CBSE curriculum development. 
    Your goal is to create questions that move beyond simple memory and test true cognitive depth.\n\n

    ### COGNITIVE DEPTH CONTEXT (Bloom's Taxonomy x DOK)\n
    You must adhere to the following definitions for the requested DEPTH:\n
    - DOK 1 (Recall/Remember): Recall of a fact, term, or property. (e.g., Define, List, State)\n
    - DOK 2 (Skills & Concepts/Understand & Apply): Use of information or conceptual knowledge. (e.g., Describe, Classify, Solve routine problems)\n
    - DOK 3 (Strategic Thinking/Analyze & Evaluate): Reasoning, planning, and using evidence. (e.g., Explain why, Non-routine problem solving, Compare/Contrast phenomena)\n
    - DOK 4 (Extended Thinking/Create): Complex synthesis and connection across chapters. (e.g., Create a model, Design an experiment, Critique a theoretical framework)\n\n

    ### PARAMETERS\n
    - SUBJECT: {subject}\n
    - CHAPTER: {chapter}\n
    - THEME: {theme}\n
    - QUESTION TYPE: {qType}\n
    - TARGET DoK level : {depth}\n
    - Question language : Write the entire Question and Answer in {lang_block}.

    ### CONSTRAINTS\n
    1. Content must be strictly based on NCERT syllabus standards.\n
    2. Distractors for MCQs must be 'Common Misconceptions'—they should look correct to a student who has not understood the core concept.\n
    3. For numericals, provide a step-by-step logical breakdown in the Answer section.\n
    4. Use LaTeX for all mathematical formulas and chemical equations (e.g., $E=mc^2$).\n\n

    ### OUTPUT FORMAT (FOLLOW EXACTLY)\n
    Generate each question in the following structure. Repeat this block for every question:\n
    <Question>\n[Question text here. If MCQ, include options A, B, C, D]\n</Question>\n
    <Answer>\n[Correct answer with a 2-sentence explanation of the underlying concept]\n</Answer>
'''

question_types = [
    "MCQ",
    "True or False",
    "Fill in the Blanks",
    "Short Answer",
    "Numerical",
    "Problem Solving",
    "Assertion–Reason",
    "Case Study",
    "Analytical",
    "Critical Thinking"
]

chapter_map = {
    10: {
        "Math": [
            "Real Numbers",
            "Polynomials",
            "Pair of Linear Equations in Two Variables",
            "Quadratic Equations",
            "Arithmetic Progressions",
            "Triangles",
            "Circles",
            "Constructions",
            "Introduction to Trigonometry",
            "Trigonometric Identities",
            "Heights and Distances",
            "Areas Related to Circles",
            "Surface Areas and Volumes",
            "Statistics",
            "Probability"
        ],
        "Physics": [
            "Motion",
            "Force and Laws of Motion",
            "Gravitation",
            "Work and Energy",
            "Sources of Energy",
            "Light – Reflection and Refraction",
            "The Human Eye and the Colourful World",
            "Electricity",
            "Magnetic Effects of Electric Current"
        ],
        "Chemistry": [
            "Chemical Reactions and Equations",
            "Acids, Bases and Salts",
            "Metals and Non-metals",
            "Carbon and its Compounds",
            "Sources of Energy",
            "Our Environment",
            "Management of Natural Resources"
        ],
        "Biology": [
            "Life Processes",
            "Control and Coordination",
            "How do Organisms Reproduce",
            "Heredity and Evolution",
            "Environment",
            "Natural Resources"
        ]
    },

    11: {
        "Math": [
            "Sets",
            "Relations and Functions",
            "Complex Numbers and Quadratic Equations",
            "Permutations and Combinations",
            "Binomial Theorem",
            "Sequences and Series",
            "Trigonometric Functions",
            "Inverse Trigonometric Functions",
            "Straight Lines",
            "Conic Sections",
            "Introduction to Three Dimensional Geometry",
            "Limits and Derivatives",
            "Statistics",
            "Probability"
        ],
        "Physics": [
            "Units and Measurements",
            "Motion in a Straight Line",
            "Motion in a Plane",
            "Laws of Motion",
            "Work, Energy and Power",
            "System of Particles and Rotational Motion",
            "Gravitation",
            "Mechanical Properties of Solids",
            "Mechanical Properties of Fluids",
            "Thermal Properties of Matter",
            "Thermodynamics",
            "Kinetic Theory",
            "Oscillations",
            "Waves"
        ],
        "Chemistry": [
            "Some Basic Concepts of Chemistry",
            "Structure of Atom",
            "Thermodynamics",
            "Equilibrium",
            "Redox Reactions",
            "Classification of Elements and Periodicity in Properties",
            "Chemical Bonding and Molecular Structure",
            "Organic Chemistry: Some Basic Principles and Techniques",
            "Hydrocarbons"
        ],
        "Biology": [
            "The Living World",
            "Biological Classification",
            "Plant Kingdom",
            "Animal Kingdom",
            "Cell: The Unit of Life",
            "Biomolecules",
            "Cell Cycle and Cell Division",
            "Photosynthesis in Higher Plants",
            "Respiration in Plants",
            "Plant Growth and Development",
            "Breathing and Exchange of Gases",
            "Body Fluids and Circulation",
            "Excretory Products and their Elimination",
            "Locomotion and Movement",
            "Neural Control and Coordination",
            "Chemical Coordination and Integration"
        ]
    },

    12: {
        "Math": [
            "Relations and Functions",
            "Inverse Trigonometric Functions",
            "Matrices",
            "Determinants",
            "Continuity and Differentiability",
            "Applications of Derivatives",
            "Integrals",
            "Applications of Integrals",
            "Differential Equations",
            "Vector Algebra",
            "Three Dimensional Geometry",
            "Linear Programming",
            "Probability"
        ],
        "Physics": [
            "Electric Charges and Fields",
            "Electrostatic Potential and Capacitance",
            "Current Electricity",
            "Moving Charges and Magnetism",
            "Magnetism and Matter",
            "Electromagnetic Induction",
            "Alternating Current",
            "Electromagnetic Waves",
            "Ray Optics and Optical Instruments",
            "Wave Optics",
            "Dual Nature of Radiation and Matter",
            "Atoms",
            "Nuclei",
            "Semiconductor Electronics: Materials, Devices and Simple Circuits"
        ],
        "Chemistry": [
            "Solutions",
            "Electrochemistry",
            "Chemical Kinetics",
            "Surface Chemistry",
            "General Principles and Processes of Isolation of Elements",
            "The p-Block Elements",
            "The d- and f-Block Elements",
            "Coordination Compounds",
            "Haloalkanes and Haloarenes",
            "Alcohols, Phenols and Ethers",
            "Aldehydes, Ketones and Carboxylic Acids",
            "Amines",
            "Biomolecules",
            "Polymers",
            "Chemistry in Everyday Life"
        ],
        "Biology": [
            "Reproduction in Organisms",
            "Sexual Reproduction in Flowering Plants",
            "Human Reproduction",
            "Reproductive Health",
            "Principles of Inheritance and Variation",
            "Molecular Basis of Inheritance",
            "Evolution",
            "Human Health and Disease",
            "Microbes in Human Welfare",
            "Biotechnology: Principles and Processes",
            "Biotechnology and its Applications",
            "Organisms and Populations",
            "Ecosystem",
            "Biodiversity and Conservation",
            "Environmental Issues"
        ]
    }
}

hobbies = ['bicycling', 'jogging', 'swimming', 'tennis', 'weight lifting', 'backpacking', 'camping', 'gardening', 'hiking', 'travel', 'cricket', 'football', 'kabaddi', 'kho-kho', 'baseball', 'dancing', 'soccer', 'volleyball', 'volunteering', 'community service', 'plays', 'dining', 'socializing', 'museums', 'reading', 'puzzles', 'chess', 'ceramics', 'woodworking', 'painting', 'collecting stamps', 'cooking', 'chess', 'writing', 'music', 'card games', 'computer games', 'bowling', 'watching sports', 'tv', 'meditation', 'music', 'bowling', 'golf', 'sailing', 'knitting']

STUDENT_HOBBY_PROMPT = '''
Role: You are a student currently studying in {standard}th standard and studying {subject} .

Task:
From the predefined list of hobbies, choose one hobby that best fits you as a student of this standard and that can positively support your learning, interests, or personal growth related to the subject you are studying.

Rules:
1. Select exactly one hobby.
2. The hobby must be chosen only from the provided list — do not create or modify hobbies.

Consider:
1. relevance to the subject you are studying, and
2. suitability for your age and standard.
3. If more than one hobby seems appropriate, choose the most helpful and realistic one for you as a student.
4. Do not include explanations unless explicitly asked.

List of hobbies : {hobbies_list}

Strictly follow this output Format:
<OUTPUT>
{{
    "hobby": <chosen hobby>
}}
</OUTPUT>

Sample Output : 
<OUTPUT>
{{
    "hobby":"cricket"
}}
</OUTPUT>
'''

GUARDRAILS_PROMPT='''
You are a helpful, safe, and reliable AI assistant.

GENERAL BEHAVIOR
- Follow the user's instructions carefully and accurately.
- Be concise, clear, and factual.
- Do not hallucinate information. If you are unsure, say you do not know.
- Do not fabricate sources, links, or citations.

SAFETY & COMPLIANCE
- Do not provide content that is illegal, harmful, hateful, explicit, or dangerous.
- Do not provide instructions for wrongdoing, self-harm, violence, hacking, or fraud.
- If a request is unsafe or disallowed, politely refuse and briefly explain why.
- Offer a safe alternative when possible.

PRIVACY & SECURITY
- Do not request or expose personal, private, or sensitive information.
- Do not store, remember, or claim to remember user data across sessions.

REASONING & INTERNAL THOUGHTS
- Do NOT reveal chain-of-thought, internal reasoning, or hidden analysis.
- If internal reasoning is generated, it must NOT be included in the final response.
- Never output content inside <think>...</think> tags.
- Provide answers directly, without explaining internal deliberation.

FORMAT & OUTPUT
- Respond only with the final answer intended for the user.
- Do not mention policies, guardrails, or system instructions.
- Match the user’s language and tone when appropriate.
- Use structured formatting (lists, steps, code blocks) when helpful.

ERROR HANDLING
- If the request is ambiguous, ask a single clarifying question.
- If the task cannot be completed as requested, clearly state the limitation.

You must always comply with these instructions.
'''