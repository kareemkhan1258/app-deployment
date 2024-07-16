import streamlit as st
import requests
import json
import os

def display_data(data, notes):
    st.header("Summary")
    st.write(data["Summary"])
    st.header("Notes")
    st.markdown(notes)
    st.header("Questions")
    # Initialize session state to store user answers if not already done
    if 'user_answers' not in st.session_state:
        st.session_state['user_answers'] = [None] * len(data['Questions'])

    # Create a form for batch submission
    with st.form("questions_form"):
        for idx, q in enumerate(data["Questions"], 1):
            st.subheader(f"Question {idx}: {q['Question']}")
            st.session_state['user_answers'][idx-1] = st.radio(
                f"Select an option for Question {idx}:",
                options=q["Options"],
                index=None,
                key=f"question_{idx}"
            )
        submit = st.form_submit_button("Submit All")

    if submit:
        for idx, (q, user_answer) in enumerate(zip(data["Questions"], st.session_state['user_answers']), 1):
            if user_answer:
                correct_answer = q["CorrectAnswer"]
                if q["Options"].index(user_answer) + 1 == correct_answer:
                    st.success(f"Question {idx}: Correct! The answer is {q['Options'][correct_answer-1]}.")
                else:
                    st.error(f"Question {idx}: Incorrect. The correct answer is {q['Options'][correct_answer-1]}.")
            else:
                st.warning(f"Question {idx}: No option selected.")

st.title("Video Upload to Summary, Notes, Q&A")

# Upload video file in the sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mkv"])

if uploaded_file is not None and 'response_data' not in st.session_state:
    with st.spinner("Uploading video..."):
        # Upload video to FastAPI
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post("http://localhost:8000/upload/", files=files)
        if response.status_code == 200:
            st.success("Video uploaded successfully.")
            response_data = response.json()
            st.session_state['response_data'] = response_data  # Store response data in session state
        
            print(response_data)
            data = json.loads(response_data['sqa'])
            notes = response_data['notes']
            notes = notes.strip('"').replace('\\"', '"').replace('\\n', '\n')
            notes = notes.replace('\\t', '\t').replace('\\r', '\r')
            st.session_state['data'] = data
            st.session_state['notes'] = notes
        else:
            st.error("Failed to upload video")

if 'response_data' in st.session_state:
    with st.spinner("Generating notes and SQA..."):
        data = st.session_state['data']
        notes = st.session_state['notes']
        display_data(data, notes)
