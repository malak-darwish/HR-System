import io
import json
import queue
import sys
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request, send_from_directory
from pypdf import PdfReader

from graph import build_interview_graph

HERE = Path(__file__).parent
JOBS = json.loads((HERE / "jobs.json").read_text(encoding="utf-8"))
JOBS_BY_ID = {job["id"]: job for job in JOBS}

ANSWER_TIMEOUT = 30 * 60   
POLL_TIMEOUT = 25         
NODES = {"screener", "interviewer", "verification", "recruiter"}
DECISION_KEYS = ("decision", "final_decision", "hiring_decision", "recommendation")

app = Flask(__name__, static_folder=str(HERE / "static"))
sessions: dict[str, "Session"] = {}


class Session:
    def __init__(self):
        self.events = queue.Queue()    
        self.answers = queue.Queue()   
        self.waiting_for_answer = False


def to_jsonable(obj):
    """Final state can hold Pydantic objects (e.g. Claim); convert everything to JSON types."""
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def print_report(result: dict, job_title: str, tag: str):
    """Print the pipeline results to the terminal (for the grader, not the candidate)."""
    line = "=" * 60
    print(f"\n{line}\n[{tag}] INTERVIEW FINISHED: {job_title}\n{line}")
    print("screening_passed:", result.get("screening_passed"), "| match_score:", result.get("match_score"))

    for key in DECISION_KEYS:
        if result.get(key) is not None:
            print(f"{key}:", result[key])
            break

    qs, ans = result.get("questions") or [], result.get("answers") or []
    print(f"\nquestions: {len(qs)} | answers: {len(ans)} | follow_up_count: {result.get('follow_up_count')}")
    for i, (q, a) in enumerate(zip(qs, ans), 1):
        a = str(a)
        print(f"\nQ{i}: {q}\nA{i}: {a[:200]}{'...' if len(a) > 200 else ''}")

    flags = result.get("consistency_flags") or {}
    print("\nconsistency_flags:", flags)
    print("verification_notes:", result.get("verification_notes"))

    print("\nclaims:")
    for c in result.get("claims") or []:
        flag = "  <-- FLAGGED" if flags.get(c.get("claim_id")) else ""
        print(f"  - {c.get('text')!r} [{c.get('category')}] "
              f"verified={c.get('verified')} conf={c.get('confidence')} source={c.get('source')}{flag}")
    print(line + "\n", flush=True)


def run_pipeline(session: Session, cv_text: str, job: dict, thread_id: str):
    tag = thread_id[:6]  

    def answer_provider(question: str) -> str:
        session.waiting_for_answer = True
        session.events.put({"type": "question", "text": question})
        try:
            return session.answers.get(timeout=ANSWER_TIMEOUT)
        except queue.Empty:
            raise TimeoutError("The candidate did not answer within 30 minutes.")
        finally:
            session.waiting_for_answer = False

    try:
        print(f"\n[{tag}] New application: {job['title']}", flush=True)
        graph = build_interview_graph(answer_provider=answer_provider)
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"cv_text": cv_text, "job_description": job["description"]}

        for chunk in graph.stream(inputs, config=config, stream_mode="updates"):
            for node in chunk:
                if node in NODES:
                    print(f"[{tag}] {node} done", flush=True)
                    session.events.put({"type": "stage", "node": node})

        final = to_jsonable(graph.get_state(config).values)
        print_report(final, job["title"], tag)
        session.events.put({"type": "done", "result": final})
    except Exception as exc:  
        print(f"[{tag}] ERROR: {type(exc).__name__}: {exc}", flush=True)
        session.events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})


def extract_cv_text() -> str:
    upload = request.files.get("cv_file")
    if upload and upload.filename:
        data = upload.read()
        if upload.filename.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return data.decode("utf-8", errors="ignore")
    return request.form.get("cv_text", "")

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/jobs")
def list_jobs():
    return jsonify(JOBS)


@app.post("/api/apply")
def apply():
    job = JOBS_BY_ID.get(request.form.get("job_id", ""))
    if not job:
        return jsonify(error="This role no longer exists. Refresh the page."), 404

    cv_text = extract_cv_text().strip()
    if len(cv_text) < 50:
        return jsonify(error="No readable text found in the CV. Upload a text-based PDF or paste your CV."), 400

    session_id = uuid.uuid4().hex
    session = Session()
    sessions[session_id] = session
    threading.Thread(
        target=run_pipeline,
        args=(session, cv_text, job, session_id),
        daemon=True,
    ).start()
    return jsonify(session_id=session_id, job_title=job["title"])


@app.get("/api/session/<session_id>/next")
def next_event(session_id):
    """Long poll: returns the next event, or {"type": "idle"} after POLL_TIMEOUT."""
    session = sessions.get(session_id)
    if not session:
        return jsonify(error="Session not found. Start a new application."), 404
    try:
        return jsonify(session.events.get(timeout=POLL_TIMEOUT))
    except queue.Empty:
        return jsonify(type="idle")


@app.post("/api/session/<session_id>/answer")
def submit_answer(session_id):
    session = sessions.get(session_id)
    if not session:
        return jsonify(error="Session not found. Start a new application."), 404
    answer = ((request.get_json(silent=True) or {}).get("answer") or "").strip()
    if not answer:
        return jsonify(error="Write an answer before sending."), 400
    if not session.waiting_for_answer:
        return jsonify(error="No question is waiting for an answer."), 409
    session.answers.put(answer)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)