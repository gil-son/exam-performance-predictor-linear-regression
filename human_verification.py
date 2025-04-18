# human_verification.py
import streamlit as st
import random

def run_human_verification():
    if "human_check_number" not in st.session_state:
        st.session_state.human_check_number = random.randint(1, 10)

    if "human_verified" not in st.session_state:
        st.session_state.human_verified = False

    if "button_clicked" not in st.session_state:
        st.session_state.button_clicked = False

    if not st.session_state.human_verified:
        st.subheader("🧠 Human Verification")
        st.write("Match the number below to prove you're not a bot:")

        # Show target number
        st.code(f"Target Number: {st.session_state.human_check_number}", language="markdown")

        # Slider for user input
        user_value = st.slider("Move the slider to match the target number", 1, 100, 1)

        # Confirm button
        if st.button("✅ Confirm Number"):
            st.session_state.button_clicked = True
            if user_value == st.session_state.human_check_number:
                st.session_state.human_verified = True
                st.success("✅ Human verification successful!")
                st.rerun()
            else:
                st.session_state.human_verified = False

        if st.session_state.button_clicked and not st.session_state.human_verified:
            st.error("❌ Incorrect number. Please try again.")

        st.stop()