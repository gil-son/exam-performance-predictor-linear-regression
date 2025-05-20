# human_verification.py
import streamlit as st
import random
import os
from datetime import datetime, timedelta

CLOCK_EMOJIS = {
    "⌛": "sand",
    "🕐": 1,
    "🕑": 2,
    "🕒": 3,
    "🕓": 4,
    "🕔" :5,
    "🕕": 6,
    "🕖": 7,
    "🕗": 8,
    "🕙": 10,
    "🕚": 11,
    "🕛" :12
}
BLOCK_DURATION_MINUTES = 5
MAX_ATTEMPTS = 3
BLOCK_FILE = "block_until.txt"

def _load_block_until():
    if os.path.exists(BLOCK_FILE):
        ts = open(BLOCK_FILE).read().strip()
        return datetime.fromisoformat(ts)
    return None

def _save_block_until(dt: datetime):
    with open(BLOCK_FILE, "w") as f:
        f.write(dt.isoformat())

def _clear_block_file():
    if os.path.exists(BLOCK_FILE):
        os.remove(BLOCK_FILE)

def run_human_verification():
    now = datetime.now()
    persisted = _load_block_until()
    if persisted and "block_until" not in st.session_state:
        st.session_state.block_until = persisted

    st.session_state.setdefault("human_verified", False)
    st.session_state.setdefault("attempts_left", MAX_ATTEMPTS)
    st.session_state.setdefault("clock_choice", None)
    if "human_target_hour" not in st.session_state:
        st.session_state.human_target_hour = random.choice(list(CLOCK_EMOJIS.values()))

    if st.session_state.get("block_until"):
        if now < st.session_state.block_until:
            rem = st.session_state.block_until - now
            secs = int(rem.total_seconds())
            st.error(f"⛔ Too many failed attempts. Try again in {secs} seconds.")
            st.stop()
        else:
            _clear_block_file()
            st.session_state.attempts_left = MAX_ATTEMPTS
            st.session_state.block_until = None
            st.session_state.human_target_hour = random.choice(list(CLOCK_EMOJIS.values()))
            st.session_state.clock_choice = None

    if not st.session_state.human_verified:
        st.subheader("🧠 Human Verification")
        target = st.session_state.human_target_hour
        st.write(f"Click the clock that shows **{target}:00**")

        # Add this CSS to make all buttons larger
        st.markdown("""
            <style>
                    
            p {
                font-size: 32px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Render clock buttons in 2 rows of 4
        emojis = list(CLOCK_EMOJIS.keys())
        random.shuffle(emojis)
        cols = st.columns(4)

        for i, emoji in enumerate(emojis):
            with cols[i % 4]:
                # Unique container for styling
                with st.container():
                    st.markdown('<div class="clock-button-container">', unsafe_allow_html=True)
                    if st.button(emoji, key=f"clock_{emoji}"):
                        st.session_state.clock_choice = emoji
                    st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.clock_choice:
            st.markdown(f"🔎 You selected: <span style='font-size: 32px'>{st.session_state.clock_choice}</span>", unsafe_allow_html=True)
            if st.button("✅ Submit"):
                if CLOCK_EMOJIS[st.session_state.clock_choice] == target:
                    st.session_state.human_verified = True
                    st.success("✅ Human verification successful!")
                    st.rerun()
                else:
                    st.session_state.attempts_left -= 1
                    if st.session_state.attempts_left <= 0:
                        unblock = now + timedelta(minutes=BLOCK_DURATION_MINUTES)
                        st.session_state.block_until = unblock
                        _save_block_until(unblock)
                        st.error("❌ Too many wrong answers — you’re blocked for 2 minutes.")
                        st.stop()
                    else:
                        st.error(f"❌ Wrong! {st.session_state.attempts_left} attempt(s) left.")
                        st.session_state.human_target_hour = random.choice(list(CLOCK_EMOJIS.values()))
                        st.session_state.clock_choice = None
                        st.rerun()

        st.stop()