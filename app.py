from flask import Flask, request, jsonify, render_template
import json
import os

app = Flask(__name__)

# Input counterfactuals
TRIAL_FILE = "/Users/pingjunhong/Desktop/UW/pragma/user_study/gpt-4/cot/gpt-3.5_cfq_clean.jsonl"
# Output responses
RESULT_FILE = "/Users/pingjunhong/Desktop/UW/pragma/user_study/user_study_responses_gpt-4-cot-gpt-3.5.jsonl"

# Load all trials into memory
TRIALS = []
if not os.path.exists(TRIAL_FILE):
    raise FileNotFoundError(f"Can't find file: {TRIAL_FILE}")

with open(TRIAL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        TRIALS.append(obj)

print(f"Loaded {len(TRIALS)} trials from {TRIAL_FILE}")

# Build a mapping from qid to trial index (to resume from last position)
QID_TO_INDEX = {}
for i, trial in enumerate(TRIALS):
    qid = trial.get("qid")
    if qid is not None:
        QID_TO_INDEX[qid] = i


def compute_resume_position():
    """
    Inspect RESULT_FILE and decide from which trial index and phase we should resume.

    Logic:
    - If RESULT_FILE does not exist or is empty -> start from index 0, phase 1.
    - Otherwise read the last non-empty line and parse it:
        - last_qid, last_phase
        - Map last_qid to index via QID_TO_INDEX.
        - If last_phase == 1: resume with same index, phase 2.
        - If last_phase == 2: resume with next index, phase 1.
    """
    if not os.path.exists(RESULT_FILE):
        return 0, 1  # start from the very beginning

    last_line = None
    try:
        with open(RESULT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line

        if last_line is None:
            # File exists but is empty / only blank lines
            return 0, 1

        last = json.loads(last_line)
        last_qid = last.get("qid")
        last_phase = int(last.get("phase", 1))

        idx = QID_TO_INDEX.get(last_qid)
        if idx is None:
            # qid not found in current trial file -> safest is to restart
            print("Warning: last qid from result file not found in trials. Starting from beginning.")
            return 0, 1

        if last_phase == 1:
            # Next: phase 2 of the same trial
            return idx, 2
        else:
            # last_phase == 2: go to next trial, phase 1
            next_idx = idx + 1
            if next_idx >= len(TRIALS):
                # All trials completed
                return len(TRIALS), 1
            return next_idx, 1

    except Exception as e:
        print(f"Warning: failed to compute resume position: {e}")
        # If anything goes wrong, fall back to start from the beginning
        return 0, 1


START_INDEX, START_PHASE = compute_resume_position()
print(f"Resume from trial index {START_INDEX}, phase {START_PHASE}")


@app.route("/")
def index():
    # Pass start_index and start_phase to the frontend so it can resume
    return render_template(
        "index.html",
        n_trials=len(TRIALS),
        start_index=START_INDEX,
        start_phase=START_PHASE,
    )


@app.route("/get_trial/<int:idx>", methods=["GET"])
def get_trial(idx):
    """
    Return the idx-th trial content, or done=True if idx is out of range.
    """
    if idx < 0 or idx >= len(TRIALS):
        return jsonify({"done": True})

    trial = TRIALS[idx]

    return jsonify({
        "done": False,
        "idx": idx,
        "qid": trial.get("qid"),
        "transform_type": trial.get("transform_type"),
        "original_question": trial.get("original_question"),
        "model_answer": trial.get("model_answer"),
        "followup_question": trial.get("followup_question"),
        "explanation": trial.get("explanation"),
    })


@app.route("/submit", methods=["POST"])
def submit():
    """
    Receive one response from the frontend and append it to RESULT_FILE.

    Expected JSON payload:
    {
      "qid": ...,
      "phase": 1 or 2,
      "transform_type": ...,
      "original_question": ...,
      "model_answer": "yes" / "no",
      "followup_question": ...,
      "explanation": null or string,
      "can_guess": true / false,
      "prediction": "yes" / "no" / "cannot_guess"
    }
    """
    data = request.get_json(force=True)

    required_keys = [
        "qid", "phase", "transform_type", "original_question",
        "model_answer", "followup_question", "explanation",
        "can_guess", "prediction"
    ]
    for k in required_keys:
        if k not in data:
            return jsonify({"status": "error", "message": f"Missing key: {k}"}), 400

    # Append the response as one JSONL line
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Use port 5001 because 5000 is already in use on your machine
    app.run(host="0.0.0.0", port=5001, debug=True)
