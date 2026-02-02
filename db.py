from tinydb import TinyDB, Query
db = TinyDB("questions_db.json")
questions_db = db.table("questions_db")
Q = Query()

def insert_question(qid, question, answer, model, metadata=None, scores=None):
    questions_db.insert({
        "qid": int(qid),
        "question": str(question),
        "answer": str(answer),
        "model": str(model),
        "metadata": metadata or {},
        "scores": scores or {}
    })

def get_by_qid(qid):
    return questions_db.get(Q.qid == qid)

def update_answer(qid, new_answer):
    questions_db.update(
        {"answer": new_answer},
        Q.qid == qid
    )

def find_by_metadata(key, value):
    return questions_db.search(Q.metadata[key] == value)

def insert_unique_question(*args, **kwargs):
    qid = kwargs.get("qid")
    if questions_db.contains(Q.qid == qid):
        raise ValueError(f"Question with qid {qid} already exists")
    insert_question(*args, **kwargs)

def get_all_questions_db():
    return questions_db.all()
