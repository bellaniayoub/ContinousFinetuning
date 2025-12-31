import streamlit as st
import os
from openai import OpenAI

# --- Configuration ---
# Allow overriding VLLM URL from environment variable
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")
MODEL_NAME = "Qwen2.5_CostumerService" # Must match --lora-modules name in docker-compose

st.set_page_config(page_title="Customer Service Bot", page_icon="🤖")

st.title("🤖 AI Customer Support Agent")
st.markdown(f"Running on: `{MODEL_NAME}`")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Settings")
    system_prompt = st.text_area(
        "System Prompt", 
        value="You are a helpful customer support assistant for an e-commerce platform. Be polite and concise."
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    reset_btn = st.button("Reset Chat")

# --- State Management ---
if "messages" not in st.session_state or reset_btn:
    st.session_state.messages = []
    if system_prompt:
        st.session_state.messages.append({"role": "system", "content": system_prompt})

# Initialize OpenAI Client (pointing to vLLM)
client = OpenAI(
    api_key="EMPTY", # vLLM doesn't require a real key by default
    base_url=VLLM_API_URL
)

# --- Chat Interface ---
# Display history (excluding system prompt)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("How can I help you today?"):
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=st.session_state.messages,
                temperature=temperature,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # 3. Save assistant message
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error communicating with vLLM: {e}")
