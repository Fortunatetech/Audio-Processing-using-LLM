import streamlit as st
import pandas as pd
import json
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=st.secrets['GEMINI_API_KEY'])
model = genai.GenerativeModel('models/gemini-2.0-flash-thinking-exp')

st.title("Wage Payment Law Analyzer 🏛️📊")

# 1. Upload Excel
uploaded_file = st.file_uploader("Upload the BNA Wage Payment Requirements Excel", type=["xlsx"])
if uploaded_file:
    # Read the specific sheet
    df = pd.read_excel(uploaded_file, sheet_name="BNA - WAGE PAYMENT REQUIREMENT")

    # Assume the first column holds "Federal" in one row
    first_col = df.columns[0]
    fed_row = df[df[first_col].str.contains("Federal", case=False, na=False)].iloc[0]

    st.subheader("🔍 Federal Guidelines (Pay Statements Analysis)")
    st.write(fed_row[["Pay Statements Analysis"]])

    # 2. User Question
    question = st.text_input("Ask your question about state deviations vs federal guidelines:")

    if question:
        # Prepare data for prompt
        federal_text = fed_row.to_dict()
        state_df = df[~df[first_col].str.contains("Federal", case=False, na=False)].copy()
        state_list = state_df.to_dict(orient="records")

        # Build prompt
        prompt = (
            "You are a legal research assistant.\n"
            "Compare state wage-payment requirements against the following federal guidelines, "
            "and identify any deviations for each state across all categories "
            "(Frequency of Payment, Method of Payment, Pay Statements, Notification Requirements, etc.).\n"
            "Federal Guidelines:\n" + json.dumps(federal_text, indent=2) + "\n"
            "State Data (list of state dicts):\n" + json.dumps(state_list, indent=2) + "\n"
            "Provide a summary of deviations by state."
        )

        # Call Gemini with file and prompt
        response = model.generate_content(
            contents=[uploaded_file, prompt],
            request_options={"timeout": 300}
        )

        st.subheader("📑 Analysis Result")
        st.write(response.text)
