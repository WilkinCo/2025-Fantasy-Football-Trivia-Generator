import os
import json
import re
import random

import pandas as pd
import streamlit as st
import anthropic

# SETTINGS
CSV_FILE = "2025ranks.csv"
MODEL_NAME = "claude-opus-4-6"
TEMPERATURE = 0.5
MAX_TOKENS = 1200
QUESTIONS_PER_ROUND = 5


def extract_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return text


def safe_json_loads(text: str) -> dict:
    cleaned = extract_json(text)
    return json.loads(cleaned)



# DATA LOADING
@st.cache_data
def load_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, header=1)

    df.columns = [
        "Rk", "Player", "Tm", "FantPos", "Age", "G", "GS",
        "PassCmp", "PassAtt", "PassYds", "PassTD", "PassInt",
        "RushAtt", "RushYds", "RushYPA", "RushTD",
        "RecTgt", "Rec", "RecYds", "RecYPR", "RecTD",
        "Fmb", "FL",
        "TotalTD", "TwoPM", "TwoPP",
        "FantPt", "PPR", "DKPt", "FDPt", "VBD", "PosRank", "OvRank", "PlayerID"
    ]

    numeric_cols = [
        "Rk", "Age", "G", "GS",
        "PassCmp", "PassAtt", "PassYds", "PassTD", "PassInt",
        "RushAtt", "RushYds", "RushYPA", "RushTD",
        "RecTgt", "Rec", "RecYds", "RecYPR", "RecTD",
        "Fmb", "FL", "TotalTD", "TwoPM", "TwoPP",
        "FantPt", "PPR", "DKPt", "FDPt", "VBD", "PosRank", "OvRank"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["PPR"].notna()]
    df = df[df["PPR"] > 0]

    df["Player"] = df["Player"].astype(str).str.replace("*", "", regex=False)
    df["Player"] = df["Player"].astype(str).str.replace("+", "", regex=False)

    df = df[df["FantPos"].isin(["QB", "RB", "WR", "TE"])].copy()
    return df


def make_fact(row: pd.Series) -> str:
    pos = row["FantPos"]
    player = row["Player"]
    team = row["Tm"]

    if pos == "QB":
        return (
            f"{player} was a {team} QB with {int(row['PassYds'])} passing yards, "
            f"{int(row['PassTD'])} passing touchdowns, {int(row['PassInt'])} interceptions, "
            f"{int(row['RushYds'])} rushing yards, {int(row['RushTD'])} rushing touchdowns, "
            f"and {float(row['PPR']):.1f} PPR points."
        )

    if pos == "RB":
        return (
            f"{player} was a {team} RB with {int(row['RushYds'])} rushing yards, "
            f"{int(row['RushTD'])} rushing touchdowns, {int(row['Rec'])} receptions, "
            f"{int(row['RecYds'])} receiving yards, {int(row['RecTD'])} receiving touchdowns, "
            f"and {float(row['PPR']):.1f} PPR points."
        )

    if pos == "WR":
        return (
            f"{player} was a {team} WR with {int(row['Rec'])} receptions, "
            f"{int(row['RecYds'])} receiving yards, {int(row['RecTD'])} receiving touchdowns, "
            f"{int(row['RushYds'])} rushing yards, and {float(row['PPR']):.1f} PPR points."
        )

    return (
        f"{player} was a {team} TE with {int(row['Rec'])} receptions, "
        f"{int(row['RecYds'])} receiving yards, {int(row['RecTD'])} receiving touchdowns, "
        f"and {float(row['PPR']):.1f} PPR points."
    )


def add_facts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Fact"] = out.apply(make_fact, axis=1)
    return out


def filter_by_difficulty(df: pd.DataFrame, difficulty: str) -> pd.DataFrame:
    if difficulty == "Easy":
        return df[df["PPR"] >= 200].copy()
    if difficulty == "Medium":
        return df[df["PPR"] >= 100].copy()
    return df[df["PPR"] >= 40].copy()


def build_context(df: pd.DataFrame, difficulty: str, position: str, n: int) -> tuple[str, pd.DataFrame]:
    filtered = filter_by_difficulty(df, difficulty)

    if position != "All":
        filtered = filtered[filtered["FantPos"] == position].copy()

    if filtered.empty:
        raise ValueError("No players matched the selected filters.")

    sample_size = min(n, len(filtered))
    sample_df = filtered.sample(n=sample_size)

    context = "\n".join(f"- {fact}" for fact in sample_df["Fact"])
    return context, sample_df


def build_prompt(context: str, difficulty: str, position: str) -> str:
    position_text = "mixed positions" if position == "All" else position

    return f"""
You are an NFL fantasy football trivia generator.

Use ONLY the facts below.
Do NOT invent players, teams, rankings, or statistics.
If a question cannot be answered directly from the facts, do not generate it.

Generate exactly {QUESTIONS_PER_ROUND} multiple-choice trivia questions.

Requirements:
- The questions should match a {difficulty.lower()} difficulty level
- The player pool is: {position_text}
- Questions must be directly answerable from the facts
- Each question must have exactly 4 answer options
- Multiple-choice options must be realistic
- Only one answer can be correct for each question
- Include the correct answer for every question
- Return ONLY valid JSON
- Do NOT include markdown fences
- Do NOT include any text before or after the JSON
- The JSON must exactly match this schema:
{{
  "multiple_choice": [
    {{
      "question": "string",
      "options": ["A", "B", "C", "D"],
      "answer": "string"
    }}
  ]
}}

Facts:
{context}
""".strip()


def get_claude_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Set it before running Streamlit.\n"
            'PowerShell example:\n$env:ANTHROPIC_API_KEY="your-key-here"'
        )
    return anthropic.Anthropic(api_key=api_key)


def normalize_trivia(payload: dict) -> dict:
    if "multiple_choice" not in payload or not isinstance(payload["multiple_choice"], list):
        raise ValueError("Claude did not return a valid multiple_choice list.")

    questions = payload["multiple_choice"]

    cleaned_questions = []
    for q in questions:
        if not isinstance(q, dict):
            continue

        question = q.get("question")
        options = q.get("options")
        answer = q.get("answer")

        if not question or not isinstance(question, str):
            continue
        if not options or not isinstance(options, list) or len(options) != 4:
            continue
        if not answer or not isinstance(answer, str):
            continue
        if answer not in options:
            continue

        cleaned_questions.append(
            {
                "question": question.strip(),
                "options": [str(opt).strip() for opt in options],
                "answer": answer.strip(),
            }
        )

    if not cleaned_questions:
        raise ValueError("No valid multiple-choice questions were returned.")

    return {"multiple_choice": cleaned_questions[:QUESTIONS_PER_ROUND]}


def generate_trivia(prompt: str) -> tuple[dict, str]:
    client = get_claude_client()

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()
    parsed = safe_json_loads(raw_text)
    normalized = normalize_trivia(parsed)
    return normalized, raw_text


def generate_round(df: pd.DataFrame, difficulty: str, position: str, num_facts: int):
    context, sample_df = build_context(df, difficulty, position, num_facts)
    prompt = build_prompt(context, difficulty, position)
    trivia, raw_output = generate_trivia(prompt)

    for q in trivia.get("multiple_choice", []):
        options = q["options"][:]
        random.shuffle(options)
        q["options"] = options

    return trivia, sample_df, raw_output



# STREAMLIT UI
st.set_page_config(page_title="NFL Fantasy Trivia Generator", page_icon="🏈", layout="centered")
st.title("NFL Fantasy Trivia Generator")

if "trivia" not in st.session_state:
    st.session_state.trivia = None
if "sample_df" not in st.session_state:
    st.session_state.sample_df = None
if "raw_output" not in st.session_state:
    st.session_state.raw_output = ""
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "game_active" not in st.session_state:
    st.session_state.game_active = False
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "questions_answered" not in st.session_state:
    st.session_state.questions_answered = 0
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "last_submitted_q" not in st.session_state:
    st.session_state.last_submitted_q = -1

with st.sidebar:
    st.header("Settings")
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    position = st.selectbox("Position", ["All", "QB", "RB", "WR", "TE"], index=0)
    num_facts = st.slider("Fact pool size", min_value=6, max_value=20, value=12, step=1)
    show_debug = st.checkbox("Show raw Claude output", value=False)

df = add_facts(load_data(CSV_FILE))

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Start Game", use_container_width=True):
        try:
            trivia, sample_df, raw_output = generate_round(df, difficulty, position, num_facts)
            st.session_state.trivia = trivia
            st.session_state.sample_df = sample_df
            st.session_state.raw_output = raw_output
            st.session_state.last_error = ""
            st.session_state.game_active = True
            st.session_state.current_q = 0
            st.session_state.score = 0
            st.session_state.questions_answered = 0
            st.session_state.feedback = None
            st.session_state.last_submitted_q = -1
            st.rerun()
        except Exception as e:
            st.session_state.last_error = str(e)

with col2:
    if st.button("Stop Game", use_container_width=True):
        st.session_state.game_active = False
        st.session_state.current_q = 0
        st.session_state.feedback = None
        st.session_state.last_submitted_q = -1
        st.rerun()

with col3:
    if st.button("Reset Score", use_container_width=True):
        st.session_state.score = 0
        st.session_state.questions_answered = 0
        st.session_state.feedback = None
        st.session_state.last_submitted_q = -1
        st.rerun()

if st.session_state.last_error:
    st.error(st.session_state.last_error)

    if show_debug and st.session_state.raw_output:
        with st.expander("Raw Claude output"):
            st.code(st.session_state.raw_output, language="json")

st.subheader("Game Stats")
s1, s2, s3 = st.columns(3)
with s1:
    st.metric("Score", st.session_state.score)
with s2:
    st.metric("Answered", st.session_state.questions_answered)
with s3:
    accuracy = (
        round((st.session_state.score / st.session_state.questions_answered) * 100, 1)
        if st.session_state.questions_answered > 0
        else 0.0
    )
    st.metric("Accuracy %", accuracy)

if st.session_state.game_active and st.session_state.trivia:
    questions = st.session_state.trivia.get("multiple_choice", [])

    if not questions:
        st.error("No questions available.")
    else:
        q_index = st.session_state.current_q

        if q_index >= len(questions):
            try:
                trivia, sample_df, raw_output = generate_round(df, difficulty, position, num_facts)
                st.session_state.trivia = trivia
                st.session_state.sample_df = sample_df
                st.session_state.raw_output = raw_output
                st.session_state.current_q = 0
                st.session_state.feedback = None
                st.session_state.last_submitted_q = -1
                st.rerun()
            except Exception as e:
                st.session_state.last_error = str(e)
                st.error(str(e))
        else:
            current_question = questions[q_index]

            st.markdown(f"### Question {st.session_state.questions_answered + 1}")
            st.write(current_question["question"])

            answer_key = f"answer_{st.session_state.questions_answered}_{q_index}"
            selected_answer = st.radio(
                "Choose an answer:",
                current_question["options"],
                key=answer_key,
                label_visibility="collapsed"
            )

            submit_col, next_col = st.columns(2)

            with submit_col:
                if st.button("Submit Answer", use_container_width=True):
                    if st.session_state.last_submitted_q != q_index:
                        is_correct = selected_answer == current_question["answer"]

                        st.session_state.questions_answered += 1
                        st.session_state.last_submitted_q = q_index

                        if is_correct:
                            st.session_state.score += 1
                            st.session_state.feedback = {
                                "type": "success",
                                "message": "Correct!"
                            }
                        else:
                            st.session_state.feedback = {
                                "type": "error",
                                "message": f"Incorrect. Correct answer: {current_question['answer']}"
                            }

                        st.rerun()

            with next_col:
                if st.button("Next Question", use_container_width=True):
                    st.session_state.current_q += 1
                    st.session_state.feedback = None
                    st.session_state.last_submitted_q = -1
                    st.rerun()

            if st.session_state.feedback:
                if st.session_state.feedback["type"] == "success":
                    st.success(st.session_state.feedback["message"])
                else:
                    st.error(st.session_state.feedback["message"])

            if show_debug and st.session_state.raw_output:
                with st.expander("Raw Claude output"):
                    st.code(st.session_state.raw_output, language="json")

else:
    st.info("Click Start Game to begin")