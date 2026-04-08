"""
==============================================
STUDENT ACADEMIC RISK PREDICTION SYSTEM
Chatbot Version — app_chatbot.py
==============================================
Run:  python app_chatbot.py
Open: http://127.0.0.1:5000
"""

from flask import Flask, render_template_string, request, jsonify, session
import joblib
import pandas as pd
import sqlite3
from datetime import datetime
import re, os

app = Flask(__name__)
app.secret_key = "student_risk_bot_2026"

# ── Load model ─────────────────────────────
model           = joblib.load("model/model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")
print("✅ Model loaded!")

# ── Chatbot conversation flow ──────────────
# Each step asks one question
STEPS = [
    {
        "key":     "name",
        "ask":     "👋 Hello! I am the **Student Risk Prediction Bot**.\n\nI will ask you a few questions about the student and predict their academic risk.\n\nLet's start! 📝\n\n**What is the student's full name?**",
        "type":    "text",
        "hint":    "Type the student name (e.g. Priya Nair)",
    },
    {
        "key":     "gender",
        "ask":     "Got it! Now tell me the student's **gender**.\n\nType **Male** or **Female**",
        "type":    "choice",
        "choices": ["Male", "Female"],
        "hint":    "Type Male or Female",
    },
    {
        "key":     "attendance_pct",
        "ask":     "What is the student's **attendance percentage**?\n\n_(Enter a number between 0 and 100)_",
        "type":    "number",
        "min":     0,
        "max":     100,
        "hint":    "e.g. 75 or 88.5",
    },
    {
        "key":     "study_hours",
        "ask":     "How many **hours per day** does the student study on average?\n\n_(Enter a number between 0 and 12)_",
        "type":    "number",
        "min":     0,
        "max":     12,
        "hint":    "e.g. 3 or 4.5",
    },
    {
        "key":     "assignments",
        "ask":     "How many **assignments** has the student submitted?\n\n_(Out of 10 total assignments)_",
        "type":    "number",
        "min":     0,
        "max":     10,
        "hint":    "e.g. 7 (out of 10)",
    },
    {
        "key":     "internal_marks",
        "ask":     "What are the student's **internal exam marks**?\n\n_(Out of 50 total marks)_",
        "type":    "number",
        "min":     0,
        "max":     50,
        "hint":    "e.g. 38 (out of 50)",
    },
    {
        "key":     "backlogs",
        "ask":     "How many **subject backlogs** does the student have?\n\n_(Number of previously failed subjects)_",
        "type":    "number",
        "min":     0,
        "max":     20,
        "hint":    "e.g. 0 or 2",
    },
]

def validate_input(step, user_input):
    """Validate user input for each step. Returns (ok, cleaned_value, error_msg)"""
    val = user_input.strip()
    if not val:
        return False, None, "Please enter a value to continue."

    if step["type"] == "text":
        if len(val) < 2:
            return False, None, "Please enter a valid name (at least 2 characters)."
        return True, val, None

    if step["type"] == "choice":
        choices_lower = [c.lower() for c in step["choices"]]
        if val.lower() in choices_lower:
            idx = choices_lower.index(val.lower())
            return True, step["choices"][idx], None
        return False, None, f"Please type one of: **{', '.join(step['choices'])}**"

    if step["type"] == "number":
        try:
            num = float(val)
            if num < step["min"] or num > step["max"]:
                return False, None, f"Please enter a number between **{step['min']}** and **{step['max']}**."
            return True, num, None
        except:
            return False, None, f"Please enter a valid number (e.g. {step['hint']})"

    return True, val, None

def make_prediction(data):
    """Run ML model and return prediction result"""
    gender_encoded = 1 if data["gender"] == "Male" else 0
    input_df = pd.DataFrame([[
        data["attendance_pct"],
        data["study_hours"],
        data["assignments"],
        data["internal_marks"],
        data["backlogs"],
        gender_encoded
    ]], columns=feature_columns)

    pred  = model.predict(input_df)[0]
    conf  = round(max(model.predict_proba(input_df)[0]) * 100, 1)

    # Save to DB
    try:
        conn = sqlite3.connect("students.db")
        conn.execute("""
            INSERT INTO predictions (student_id, predicted_label, confidence, predicted_on)
            VALUES (?, ?, ?, ?)
        """, (1, int(pred), conf, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
    except:
        pass

    return int(pred), conf

def build_result_message(data, pred, conf):
    """Build the final result message for the chatbot"""
    name = data["name"]

    summary = (
        f"📋 **Summary for {name}:**\n"
        f"• Gender: {data['gender']}\n"
        f"• Attendance: {data['attendance_pct']}%\n"
        f"• Study Hours: {data['study_hours']} hrs/day\n"
        f"• Assignments: {data['assignments']}/10\n"
        f"• Internal Marks: {data['internal_marks']}/50\n"
        f"• Backlogs: {data['backlogs']}\n\n"
    )

    if pred == 1:
        result = (
            f"⚠️ **PREDICTION RESULT**\n\n"
            f"**{name} is AT RISK** of academic failure.\n"
            f"Confidence: **{conf}%**\n\n"
            f"📌 **Recommended Actions:**\n"
            f"• Counsel the student immediately\n"
            f"• Monitor attendance closely\n"
            f"• Assign additional study support\n"
            f"• Inform parents about the situation\n\n"
            f"---\nType **new** to check another student or **history** to see past predictions."
        )
    else:
        result = (
            f"✅ **PREDICTION RESULT**\n\n"
            f"**{name} is NOT AT RISK.** 🎉\n"
            f"Confidence: **{conf}%**\n\n"
            f"📌 **Recommendation:**\n"
            f"• Student is performing well\n"
            f"• Continue regular monitoring\n"
            f"• Encourage consistent study habits\n\n"
            f"---\nType **new** to check another student or **history** to see past predictions."
        )

    return summary + result

def get_history_message():
    """Fetch last 5 predictions from DB"""
    try:
        conn = sqlite3.connect("students.db")
        rows = conn.execute(
            "SELECT predicted_label, confidence, predicted_on FROM predictions ORDER BY pred_id DESC LIMIT 5"
        ).fetchall()
        conn.close()
        if not rows:
            return "No predictions yet. Type **new** to start!"
        msg = "📋 **Last 5 Predictions:**\n\n"
        for i, (label, conf, dt) in enumerate(rows, 1):
            icon  = "⚠️ AT RISK" if label == 1 else "✅ SAFE"
            msg  += f"{i}. {icon} — {conf}% confidence — {dt}\n"
        msg += "\nType **new** to check another student."
        return msg
    except:
        return "Could not fetch history. Type **new** to start!"

# ── HTML Template ──────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Student Risk Prediction Bot</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Segoe UI",sans-serif;background:#1a1a2e;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:1rem}
.chat-wrapper{width:100%;max-width:720px;display:flex;flex-direction:column;height:92vh}
.chat-header{background:linear-gradient(135deg,#1F3864,#2E75B6);border-radius:16px 16px 0 0;padding:1rem 1.5rem;display:flex;align-items:center;gap:1rem}
.bot-avatar{width:48px;height:48px;background:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.5rem}
.bot-info h2{color:#fff;font-size:1.1rem}
.bot-info p{color:#90caf9;font-size:.8rem}
.online-dot{width:10px;height:10px;background:#4caf50;border-radius:50%;margin-left:auto}
.chat-body{flex:1;background:#f0f4f8;overflow-y:auto;padding:1.5rem;display:flex;flex-direction:column;gap:1rem}
.msg{display:flex;gap:.8rem;align-items:flex-end;max-width:85%}
.msg.bot{align-self:flex-start}
.msg.user{align-self:flex-end;flex-direction:row-reverse}
.avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0}
.bot .avatar{background:#1F3864;color:#fff}
.user .avatar{background:#2E75B6;color:#fff}
.bubble{padding:.75rem 1rem;border-radius:16px;font-size:.92rem;line-height:1.6;max-width:100%}
.bot .bubble{background:#fff;color:#333;border-bottom-left-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.user .bubble{background:#2E75B6;color:#fff;border-bottom-right-radius:4px}
.bubble strong{font-weight:700}
.bubble.risk{background:#fff3f3;border-left:4px solid #c62828}
.bubble.safe{background:#f3fff3;border-left:4px solid #2e7d32}
.hint{font-size:.75rem;color:#888;margin-top:.3rem;padding-left:44px}
.quick-btns{display:flex;flex-wrap:wrap;gap:.5rem;padding-left:44px}
.quick-btn{padding:.4rem .9rem;border:1.5px solid #2E75B6;border-radius:20px;background:#fff;color:#2E75B6;font-size:.82rem;cursor:pointer;transition:all .2s}
.quick-btn:hover{background:#2E75B6;color:#fff}
.typing{display:flex;gap:4px;align-items:center;padding:.75rem 1rem;background:#fff;border-radius:16px;width:fit-content}
.dot{width:8px;height:8px;background:#2E75B6;border-radius:50%;animation:bounce .9s infinite}
.dot:nth-child(2){animation-delay:.2s}
.dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-8px)}}
.chat-input{background:#fff;border-radius:0 0 16px 16px;padding:1rem 1.5rem;display:flex;gap:.8rem;align-items:center;border-top:1px solid #e0e0e0}
.chat-input input{flex:1;border:1.5px solid #ddd;border-radius:25px;padding:.6rem 1.2rem;font-size:.95rem;outline:none;font-family:"Segoe UI",sans-serif;transition:border-color .2s}
.chat-input input:focus{border-color:#2E75B6}
.send-btn{width:42px;height:42px;border:none;border-radius:50%;background:#1F3864;color:#fff;font-size:1.1rem;cursor:pointer;transition:background .2s;display:flex;align-items:center;justify-content:center}
.send-btn:hover{background:#2E75B6}
.progress{font-size:.75rem;color:#888;text-align:center;padding:.4rem;background:#f0f4f8}
</style>
</head>
<body>
<div class="chat-wrapper">
  <div class="chat-header">
    <div class="bot-avatar">🎓</div>
    <div class="bot-info">
      <h2>Student Risk Prediction Bot</h2>
      <p>Dr. M.G.R University | MCA Project | Vishwan B S</p>
    </div>
    <div class="online-dot"></div>
  </div>
  <div class="chat-body" id="chatBody"></div>
  <div class="progress" id="progressBar"></div>
  <div class="chat-input">
    <input type="text" id="userInput" placeholder="Type your answer here..." autocomplete="off"/>
    <button class="send-btn" onclick="sendMessage()">➤</button>
  </div>
</div>

<script>
let currentStep = -1;
let collected   = {};

const STEPS = """ + str([
    {"key": s["key"], "type": s["type"],
     "choices": s.get("choices", []), "hint": s.get("hint", "")}
    for s in STEPS
]) + """;

function formatText(text) {
  return text
    .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/---/g, '<hr style="border:none;border-top:1px solid #ddd;margin:.5rem 0">')
    .replace(/\\n/g, '<br>');
}

function addBotMsg(text, extraClass='') {
  const body = document.getElementById('chatBody');
  const div  = document.createElement('div');
  div.className = 'msg bot';
  div.innerHTML = `
    <div class="avatar">🎓</div>
    <div class="bubble ${extraClass}">${formatText(text)}</div>`;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
}

function addUserMsg(text) {
  const body = document.getElementById('chatBody');
  const div  = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `
    <div class="bubble">${text}</div>
    <div class="avatar">👤</div>`;
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
}

function addHint(hint, choices=[]) {
  const body = document.getElementById('chatBody');
  if (choices.length > 0) {
    const div = document.createElement('div');
    div.className = 'quick-btns';
    choices.forEach(c => {
      const btn = document.createElement('button');
      btn.className = 'quick-btn';
      btn.textContent = c;
      btn.onclick = () => {
        document.getElementById('userInput').value = c;
        sendMessage();
      };
      div.appendChild(btn);
    });
    body.appendChild(div);
  } else if (hint) {
    const div = document.createElement('div');
    div.className = 'hint';
    div.textContent = '💡 ' + hint;
    body.appendChild(div);
  }
  body.scrollTop = body.scrollHeight;
}

function updateProgress() {
  const bar = document.getElementById('progressBar');
  if (currentStep < 0 || currentStep >= STEPS.length) { bar.textContent = ''; return; }
  bar.textContent = `Step ${currentStep+1} of ${STEPS.length} — ${Math.round((currentStep/STEPS.length)*100)}% complete`;
}

function showTyping(cb, delay=800) {
  const body = document.getElementById('chatBody');
  const div  = document.createElement('div');
  div.className = 'msg bot';
  div.id = 'typing';
  div.innerHTML = '<div class="avatar">🎓</div><div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
  setTimeout(() => {
    const t = document.getElementById('typing');
    if (t) t.remove();
    cb();
  }, delay);
}

async function sendMessage() {
  const input = document.getElementById('userInput');
  const text  = input.value.trim();
  if (!text) return;
  input.value = '';
  addUserMsg(text);

  // Special commands
  if (text.toLowerCase() === 'new') {
    currentStep = -1; collected = {};
    showTyping(() => startConversation());
    return;
  }
  if (text.toLowerCase() === 'history') {
    showTyping(async () => {
      const r = await fetch('/api/history');
      const d = await r.json();
      addBotMsg(d.message);
    });
    return;
  }
  if (text.toLowerCase() === 'help') {
    showTyping(() => addBotMsg("**Commands:**\\n• **new** — Start a new prediction\\n• **history** — See past predictions\\n• **help** — Show this message"));
    return;
  }

  if (currentStep < 0) { showTyping(() => startConversation()); return; }

  // Validate
  const r = await fetch('/api/validate', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({step: currentStep, value: text})
  });
  const d = await r.json();

  if (!d.ok) {
    showTyping(() => addBotMsg('❌ ' + d.error), 500);
    return;
  }

  collected[STEPS[currentStep].key] = d.value;
  currentStep++;
  updateProgress();

  if (currentStep >= STEPS.length) {
    // All data collected — predict
    showTyping(async () => {
      const pr = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(collected)
      });
      const pd = await pr.json();
      const cls = pd.pred === 1 ? 'risk' : 'safe';
      addBotMsg(pd.message, cls);
      currentStep = -1; collected = {};
      updateProgress();
    }, 1200);
  } else {
    showTyping(() => askStep(currentStep));
  }
}

function askStep(idx) {
  const step = STEPS[idx];
  const questions = """ + str([s["ask"] for s in STEPS]) + """;
  addBotMsg(questions[idx]);
  addHint(step.hint || '', step.choices || []);
}

function startConversation() {
  currentStep = 0;
  askStep(0);
  updateProgress();
}

// Enter key
document.getElementById('userInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendMessage();
});

// Start on load
window.onload = () => {
  showTyping(() => startConversation(), 600);
};
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/api/validate', methods=['POST'])
def api_validate():
    data     = request.json
    step_idx = data.get('step', 0)
    value    = data.get('value', '')
    if step_idx < 0 or step_idx >= len(STEPS):
        return jsonify({"ok": False, "error": "Invalid step"})
    ok, cleaned, err = validate_input(STEPS[step_idx], value)
    if ok:
        return jsonify({"ok": True, "value": cleaned})
    return jsonify({"ok": False, "error": err})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data          = request.json
    pred, conf    = make_prediction(data)
    message       = build_result_message(data, pred, conf)
    return jsonify({"pred": pred, "conf": conf, "message": message})

@app.route('/api/history')
def api_history():
    return jsonify({"message": get_history_message()})

if __name__ == '__main__':
    print("\n🤖 Student Risk Prediction CHATBOT starting...")
    print("   Open browser: http://127.0.0.1:5000\n")
    app.run(debug=True)
