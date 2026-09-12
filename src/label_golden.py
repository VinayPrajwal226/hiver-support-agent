import streamlit as st
import pandas as pd
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath('.'))

from src.intent_definitions import INTENTS
from src.ai_assistant import generate_ai_proposal, get_api_key

st.set_page_config(page_title="AI-Assisted Human Labeling Workflow", layout="wide")

INPUT_PATH = "data/golden/golden_candidates.csv"
OUTPUT_PATH = "data/golden/golden_set.csv"

ALLOWED_INTENTS = list(INTENTS.keys())
CONFIDENCE_OPTIONS = ["High", "Medium", "Low"]

EXPECTED_COLUMNS = [
    'tweet_id', 'parent_text', 'customer_text', 'intent', 'escalation',
    'escalation_reason', 'label_status', 'labeler', 'label_timestamp',
    'confidence', 'notes', 'ai_intent', 'ai_escalation',
    'ai_escalation_reason', 'ai_confidence'
]


def _load_candidates():
    """Load the authoritative golden candidate IDs. Never modified."""
    df = pd.read_csv(INPUT_PATH, dtype=str, encoding='utf-8')
    df['tweet_id'] = df['tweet_id'].astype(str).str.strip()
    return df


def _sanitize_text(value):
    """Strip newlines/carriage returns from text fields to prevent CSV row splits."""
    if value is None:
        return ""
    return str(value).replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')


def load_data():
    """
    Load the golden set, enforcing strict integrity constraints:
    - Only rows whose tweet_id is in golden_candidates are kept.
    - Candidate order is preserved (merged from candidates).
    - Extra rogue rows are silently dropped with a warning.
    - Row count is verified to be exactly 250.
    - label_status is NEVER auto-promoted from AI fields.
      Only an explicit human save via 'Approve & Save Human Label' may
      set label_status = 'human_reviewed'.
    """
    df_cand = _load_candidates()
    cand_ids = set(df_cand['tweet_id'].tolist())

    if os.path.exists(OUTPUT_PATH):
        df_saved = pd.read_csv(OUTPUT_PATH, dtype=str, encoding='utf-8')
        df_saved['tweet_id'] = df_saved['tweet_id'].astype(str).str.strip()

        # ── Integrity check: drop any rogue rows not in candidate set ──────────
        rogue_mask = ~df_saved['tweet_id'].isin(cand_ids)
        rogue_count = rogue_mask.sum()
        if rogue_count > 0:
            st.warning(
                f"⚠️ Data-integrity warning: {rogue_count} row(s) with unknown "
                f"tweet_id were found in {OUTPUT_PATH} and have been ignored. "
                f"Rogue IDs: {df_saved.loc[rogue_mask, 'tweet_id'].tolist()}"
            )
            df_saved = df_saved[~rogue_mask].copy()

        # ── Merge saved data onto candidate scaffold (preserves candidate order) ─
        df = df_cand[['tweet_id', 'parent_text', 'customer_text']].copy()
        df = df.merge(
            df_saved.drop(columns=['parent_text', 'customer_text'], errors='ignore'),
            on='tweet_id',
            how='left'
        )
    else:
        # No saved file yet — start fresh from candidates
        df = df_cand[['tweet_id', 'parent_text', 'customer_text']].copy()

    # ── Ensure all expected columns exist ────────────────────────────────────
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)

    df = df.fillna("")

    # ── Set default label_status for rows that have never been saved ─────────
    # IMPORTANT: We do NOT auto-promote based on intent/escalation fields.
    # The only way label_status becomes 'human_reviewed' is via save_row().
    for i in range(len(df)):
        status = str(df.at[i, 'label_status']).strip()
        if status not in ('human_reviewed', 'unreviewed'):
            df.at[i, 'label_status'] = 'unreviewed'

    # ── Final row-count guard ─────────────────────────────────────────────────
    if len(df) != 250:
        st.error(
            f"🚨 Critical integrity error: expected 250 rows after loading but got {len(df)}. "
            "Please contact the administrator."
        )
        st.stop()

    return df


def save_data(df):
    """
    Write the golden set to disk with integrity checks.
    Raises ValueError if the data would violate any constraint.
    """
    df_cand = _load_candidates()
    cand_ids = set(df_cand['tweet_id'].tolist())

    # Check row count
    if len(df) != 250:
        raise ValueError(f"Cannot save: expected 250 rows but df has {len(df)} rows.")

    # Check for rogue IDs
    rogue = [tid for tid in df['tweet_id'].tolist() if str(tid).strip() not in cand_ids]
    if rogue:
        raise ValueError(f"Cannot save: found non-candidate tweet_ids: {rogue}")

    # Check for duplicates
    dupes = df['tweet_id'].duplicated().sum()
    if dupes > 0:
        raise ValueError(f"Cannot save: found {dupes} duplicate tweet_id(s).")

    # Sanitize free-text fields to prevent CSV row splits
    for col in ['notes', 'escalation_reason', 'parent_text', 'customer_text']:
        if col in df.columns:
            df[col] = df[col].apply(_sanitize_text)

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')


def save_row(df, idx, intent, escalation, escalation_reason,
             confidence, notes, labeler):
    """
    Update a single candidate row and persist to disk.
    This is the ONLY function that may set label_status = 'human_reviewed'.
    """
    # Verify the tweet_id at this index is a known candidate
    df_cand = _load_candidates()
    cand_ids = set(df_cand['tweet_id'].tolist())
    tweet_id = str(df.at[idx, 'tweet_id']).strip()
    if tweet_id not in cand_ids:
        raise ValueError(
            f"Row {idx} has tweet_id='{tweet_id}' which is not in golden_candidates. Refusing to save."
        )

    df.at[idx, 'intent']            = intent
    df.at[idx, 'escalation']        = escalation
    df.at[idx, 'escalation_reason'] = _sanitize_text(escalation_reason.strip())
    df.at[idx, 'confidence']        = confidence
    df.at[idx, 'notes']             = _sanitize_text(notes.strip())
    df.at[idx, 'labeler']           = labeler.strip() if labeler.strip() else "human"
    df.at[idx, 'label_timestamp']   = datetime.datetime.now(datetime.timezone.utc).isoformat()
    df.at[idx, 'label_status']      = "human_reviewed"   # ← ONLY set here

    save_data(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Application start
# ─────────────────────────────────────────────────────────────────────────────
df = load_data()

# Calculate progress
unreviewed_mask    = df['label_status'] != "human_reviewed"
unreviewed_indices = df[unreviewed_mask].index.tolist()
first_unreviewed_idx = unreviewed_indices[0] if unreviewed_indices else 0

total_examples       = len(df)          # always 250
human_reviewed_count = total_examples - len(unreviewed_indices)
unreviewed_count     = len(unreviewed_indices)

# Session state initialization
if 'initialized' not in st.session_state or 'current_idx' not in st.session_state:
    st.session_state.current_idx = first_unreviewed_idx
    st.session_state.initialized = True

current_idx = st.session_state.current_idx
if current_idx >= total_examples:
    current_idx = total_examples - 1
    st.session_state.current_idx = current_idx

st.title("Amazon Support Dataset - AI-Assisted Human Labeling Workflow")

# API Key status warning banner
key_name, api_key = get_api_key()
if not api_key:
    st.warning(
        "**API Key Configuration:** No `OPENAI_API_KEY` or `GEMINI_API_KEY` detected in environment variables or Streamlit secrets. "
        "The workflow will use the built-in intent definition heuristic to generate AI proposals. "
        "To enable LLM generation, set `export OPENAI_API_KEY='your-key'` or `export GEMINI_API_KEY='your-key'` before running Streamlit."
    )

st.progress(human_reviewed_count / total_examples if total_examples > 0 else 0.0)
st.write(f"**Progress:** {human_reviewed_count}/{total_examples} human-reviewed ({unreviewed_count} remaining)")

if unreviewed_count == 0:
    st.success("All 250 examples have been human-reviewed and saved to the golden set!")
    st.balloons()

if total_examples > 0:
    # Action / Navigation bar
    col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns([1, 1, 1, 1, 1])
    with col_nav1:
        if st.button("Previous Example", disabled=current_idx == 0):
            st.session_state.current_idx -= 1
            st.rerun()
    with col_nav2:
        if st.button("Next Example", disabled=current_idx == total_examples - 1):
            st.session_state.current_idx += 1
            st.rerun()
    with col_nav3:
        if st.button("First Unreviewed"):
            st.session_state.current_idx = first_unreviewed_idx
            st.rerun()
    with col_nav4:
        if st.button("Generate Suggestions for All Unreviewed"):
            with st.spinner("Generating AI proposals for all unreviewed examples..."):
                for i in range(len(df)):
                    if df.at[i, 'label_status'] != "human_reviewed":
                        parent = str(df.at[i, 'parent_text'])
                        cust   = str(df.at[i, 'customer_text'])
                        _, prop, _ = generate_ai_proposal(parent, cust)
                        df.at[i, 'ai_intent']            = prop['intent']
                        df.at[i, 'ai_escalation']        = prop['escalation']
                        df.at[i, 'ai_escalation_reason'] = prop['escalation_reason']
                        df.at[i, 'ai_confidence']        = prop['confidence']
                        # label_status STAYS 'unreviewed' — AI suggestions do NOT count
                try:
                    save_data(df)
                    st.success("Generated AI proposals for all unreviewed examples! (Status remains unreviewed until approved).")
                except ValueError as e:
                    st.error(f"Save blocked by integrity check: {e}")
                st.rerun()
    with col_nav5:
        st.markdown(f"**Example {current_idx + 1} of {total_examples}**")

    st.divider()

    row = df.iloc[current_idx]
    is_human_reviewed = row['label_status'] == "human_reviewed"

    st.subheader(f"Example {current_idx + 1} of {total_examples} (Tweet ID: {row['tweet_id']})")

    col_txt1, col_txt2 = st.columns(2)
    with col_txt1:
        st.markdown("### Previous Amazon Response:")
        parent_text = str(row['parent_text']).strip() if pd.notna(row['parent_text']) and str(row['parent_text']).strip() else "*<No previous context / Blank>*"
        st.info(parent_text)
    with col_txt2:
        st.markdown("### Customer Message:")
        customer_text = str(row['customer_text']).strip() if pd.notna(row['customer_text']) else ""
        st.warning(customer_text)

    # Automatically generate AI proposal if not present for an unreviewed example
    ai_intent_val    = str(row['ai_intent']).strip()
    ai_escalation_val = str(row['ai_escalation']).strip()
    ai_reason_val    = str(row['ai_escalation_reason']).strip()
    ai_conf_val      = str(row['ai_confidence']).strip()

    if not ai_intent_val and not is_human_reviewed:
        _, prop, _ = generate_ai_proposal(row['parent_text'], row['customer_text'])
        df.at[current_idx, 'ai_intent']            = prop['intent']
        df.at[current_idx, 'ai_escalation']        = prop['escalation']
        df.at[current_idx, 'ai_escalation_reason'] = prop['escalation_reason']
        df.at[current_idx, 'ai_confidence']        = prop['confidence']
        # label_status is NOT changed here — remains 'unreviewed'
        try:
            save_data(df)
        except ValueError as e:
            st.error(f"Save blocked by integrity check: {e}")
        row           = df.iloc[current_idx]
        ai_intent_val    = str(row['ai_intent']).strip()
        ai_escalation_val = str(row['ai_escalation']).strip()
        ai_reason_val    = str(row['ai_escalation_reason']).strip()
        ai_conf_val      = str(row['ai_confidence']).strip()

    st.divider()

    # AI Suggested Label Box
    st.markdown("## AI Suggested Label")
    if ai_intent_val:
        st.info(
            f"**Proposed Intent:** `{ai_intent_val}` | "
            f"**Escalate?:** `{ai_escalation_val}` | "
            f"**Confidence:** `{ai_conf_val}`\n\n"
            f"**Escalation Reason:** {ai_reason_val if ai_reason_val else '*(None)*'}"
        )
    else:
        st.write("*(No AI proposal generated yet)*")

    st.divider()

    # Pre-fill form values
    if is_human_reviewed:
        default_intent      = str(row['intent']).strip()
        default_escalation  = str(row['escalation']).strip().lower()
        default_reason      = str(row['escalation_reason']).strip()
        default_confidence  = str(row['confidence']).strip()
        default_notes       = str(row['notes']).strip()
        default_labeler     = str(row['labeler']).strip() if row['labeler'] else "human"
    else:
        default_intent      = ai_intent_val
        default_escalation  = ai_escalation_val.lower()
        default_reason      = ai_reason_val
        default_confidence  = ai_conf_val if ai_conf_val in CONFIDENCE_OPTIONS else "High"
        default_notes       = ""   # No longer pre-filling notes with AI proposal text
        default_labeler     = "human"

    intent_idx     = ALLOWED_INTENTS.index(default_intent) if default_intent in ALLOWED_INTENTS else 0
    escalation_idx = ["yes", "no"].index(default_escalation) if default_escalation in ["yes", "no"] else 0
    confidence_idx = CONFIDENCE_OPTIONS.index(default_confidence) if default_confidence in CONFIDENCE_OPTIONS else 0

    st.markdown("## Human Review & Approval")
    if is_human_reviewed:
        st.success(f"Status: **human_reviewed** (Labeler: `{default_labeler}` at `{row['label_timestamp']}`)")
    else:
        st.write("Status: **unreviewed** (Review AI suggestions below and click Approve)")

    with st.form(key=f"review_form_{current_idx}"):
        selected_intent = st.radio(
            "Intent",
            options=ALLOWED_INTENTS,
            index=intent_idx,
            help="Review or modify the suggested support intent."
        )

        selected_escalation = st.radio(
            "Escalate to human?",
            options=["yes", "no"],
            index=escalation_idx,
            help="Should this interaction be escalated to a human agent?"
        )

        escalation_reason = st.text_input(
            "Escalation Reason (required)",
            value=default_reason,
            help="Required if escalation is 'yes'."
        )

        selected_confidence = st.selectbox(
            "Confidence",
            options=CONFIDENCE_OPTIONS,
            index=confidence_idx,
            help="Select your confidence level in this golden label."
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            labeler = st.text_input("Labeler Name", value=default_labeler)
        with col_opt2:
            notes = st.text_area("Notes", value=default_notes)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_button = st.form_submit_button(label="Approve & Save Human Label")
        with col_btn2:
            regen_button = st.form_submit_button(label="Regenerate AI Suggestion")

        if regen_button:
            _, new_prop, msg = generate_ai_proposal(row['parent_text'], row['customer_text'])
            df.at[current_idx, 'ai_intent']            = new_prop['intent']
            df.at[current_idx, 'ai_escalation']        = new_prop['escalation']
            df.at[current_idx, 'ai_escalation_reason'] = new_prop['escalation_reason']
            df.at[current_idx, 'ai_confidence']        = new_prop['confidence']
            # label_status is NOT changed — remains whatever it was
            try:
                save_data(df)
                st.success(f"Regenerated AI suggestion: {msg}")
            except ValueError as e:
                st.error(f"Save blocked by integrity check: {e}")
            st.rerun()

        if submit_button:
            validation_passed = True

            if selected_intent not in ALLOWED_INTENTS:
                st.error("Please select a valid Intent.")
                validation_passed = False

            if selected_escalation not in ["yes", "no"]:
                st.error("Please select whether to escalate to human (yes or no).")
                validation_passed = False

            if selected_confidence not in CONFIDENCE_OPTIONS:
                st.error("Please select a valid Confidence level.")
                validation_passed = False

            if selected_escalation == 'yes' and not escalation_reason.strip():
                st.error("Escalation Reason is required when escalation is yes.")
                validation_passed = False

            if validation_passed:
                try:
                    df = save_row(
                        df,
                        current_idx,
                        selected_intent,
                        selected_escalation,
                        escalation_reason,
                        selected_confidence,
                        notes,
                        labeler
                    )
                    st.success(f"✅ Saved human label for tweet_id={df.at[current_idx, 'tweet_id']}")
                except ValueError as e:
                    st.error(f"Save blocked by integrity check: {e}")
                    st.stop()

                # Advance to next unreviewed example
                new_unreviewed_mask    = df['label_status'] != "human_reviewed"
                new_unreviewed_indices = df[new_unreviewed_mask].index.tolist()

                if new_unreviewed_indices:
                    next_indices = [i for i in new_unreviewed_indices if i > current_idx]
                    st.session_state.current_idx = next_indices[0] if next_indices else new_unreviewed_indices[0]
                else:
                    if current_idx < total_examples - 1:
                        st.session_state.current_idx += 1

                st.rerun()