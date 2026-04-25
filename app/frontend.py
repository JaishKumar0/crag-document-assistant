import streamlit as st
import requests
import uuid

# --- Configuration ---
# Must match the port where your FastAPI (uvicorn) server is running
API_URL = "http://localhost:8000"

st.set_page_config(page_title="CRAG Document Assistant", page_icon="🤖", layout="wide")

# --- Session State Initialization ---
# Generates a unique session ID for LangGraph memory tracking
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "file_id" not in st.session_state:
    st.session_state["file_id"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("🤖 CRAG Document Assistant")

# --- Sidebar: Upload & Management ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or DOCX", type=["pdf", "txt", "docx"])
    
    if st.button("Upload & Index"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                # Prepare file for FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.session_state["file_id"] = response.json().get("file_id")
                        # Clear previous chat history when a new file is uploaded
                        st.session_state["messages"] = [] 
                        st.success("File successfully indexed!")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is FastAPI running?")
        else:
            st.warning("Please select a file first.")

    st.divider()
    
    if st.session_state["file_id"]:
        st.header("Document Status")
        st.info("System Ready for Queries")
        st.caption(f"Active File ID: `{st.session_state['file_id'][:8]}...`")
        
        if st.button("Delete Document Data", type="primary"):
            res = requests.delete(f"{API_URL}/delete/{st.session_state['file_id']}")
            if res.status_code == 200:
                st.session_state["file_id"] = None
                st.session_state["messages"] = []
                st.success("Vector store deleted.")
                st.rerun()
            else:
                st.error("Deletion failed.")

# --- Main Window: Chat Interface ---
if not st.session_state["file_id"]:
    st.info("👈 Please upload a document in the sidebar to begin.")
else:
    # Render existing chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Box
    if prompt := st.chat_input("Ask a question about the uploaded document..."):
        # Display user prompt immediately
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call FastAPI backend
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context & generating response..."):
                payload = {
                    "question": prompt,
                    "file_id": st.session_state["file_id"],
                    "session_id": st.session_state["session_id"]
                }
                
                try:
                    response = requests.post(f"{API_URL}/ask", json=payload)
                    
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No response generated.")
                        st.markdown(answer)
                        # Save to frontend state
                        st.session_state["messages"].append({"role": "assistant", "content": answer})
                    else:
                        error_msg = f"Backend Error: {response.json().get('detail', response.text)}"
                        st.error(error_msg)
                except requests.exceptions.ConnectionError:
                    st.error("Connection lost. Ensure the FastAPI backend is still running.")