import datetime
import streamlit as st
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# بارگذاری بانک سوالات
with open("questionBank_real.json", "r", encoding="utf-8") as f:
    data_store = json.load(f)

@st.cache_resource
def setup_ai_model():
    tokenizer_instance = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    ai_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer_instance, ai_model

tokenizer_instance, ai_model = setup_ai_model()

def get_sentence_vector(model_output, attention_mask):
    token_vectors = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_vectors.size()).float()
    sum_vectors = torch.sum(token_vectors * mask_expanded, dim=1)
    mask_sum = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_vectors / mask_sum

def find_best_response(query):
    encoded_input = tokenizer_instance([query], padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        model_result = ai_model(**encoded_input)
    
    query_vector = get_sentence_vector(model_result, encoded_input["attention_mask"])
    query_vector = F.normalize(query_vector, p=2, dim=1)
    query_array = query_vector.numpy()

    highest_similarity = 0.0
    selected_response = "I couldn't find an answer to that question."

    for question_item in data_store["questions"]:
        stored_vector = np.array(question_item["embedding"])
        similarity_value = np.dot(query_array.ravel(), stored_vector.ravel())
        
        if similarity_value > highest_similarity:
            highest_similarity = similarity_value
            selected_response = question_item["answer"]

    if highest_similarity <= 0.4:
        selected_response = "I'm not sure about that. Could you please rephrase your question?"

    return selected_response

def display_response_gradually(response_text):
    for word in response_text.split():
        yield word + " "

# --- رابط کاربری ---
st.title("🤖 AI Chat Assistant")

# تاریخچه گفتگو
if "message_history" not in st.session_state:
    st.session_state.message_history = []
    welcome_msg = "Hello! I'm your AI assistant. How can I help you today?"
    st.session_state.message_history.append({
        "sender": "assistant", 
        "text": welcome_msg,
        "display_name": "AI Assistant",
        "time": datetime.datetime.now().strftime("%H:%M")
    })

# نمایش تاریخچه با اسم
for message in st.session_state.message_history:
    # تنظیم آواتار و استایل اسم
    if message["sender"] == "user":
        avatar_icon = "👤"
        name_class = "user-name"
        display_name = message.get("display_name", "You")
    else:
        avatar_icon = "💻"
        name_class = "assistant-name"
        display_name = message.get("display_name", "AI Assistant")
    
    with st.chat_message(message["sender"], avatar=avatar_icon):
        # نمایش اسم و زمان
        st.markdown(f"""
        <div class="message-header">
            <div class="{name_class}">{display_name}</div>
            <div class="time-stamp">
            {message.get('time', '')}
        </div>
        """, unsafe_allow_html=True)
        
        # نمایش متن پیام
        st.markdown(message["text"])

# دریافت ورودی کاربر
user_query = st.chat_input("Type your message here...")

if user_query:
    current_time = datetime.datetime.now().strftime("%H:%M")
    
    # ذخیره پیام کاربر
    st.session_state.message_history.append({
        "sender": "user", 
        "text": user_query,
        "time": current_time,
        "display_name": "You"
    })
    
    # نمایش پیام کاربر
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"""
        <div class="message-header">
            <div class="user-name">You</div>
            <div class="time-stamp">{current_time}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(user_query)

    # تولید پاسخ
    with st.chat_message("assistant", avatar="🤖"):
        # هدر پاسخ
        st.markdown(f"""
        <div class="message-header">
            <div class="assistant-name">AI Assistant</div>
            <div class="time-stamp">{current_time}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # پیدا کردن و نمایش پاسخ
        bot_response_text = find_best_response(user_query)
        response_display = st.write_stream(display_response_gradually(bot_response_text))

    # ذخیره پاسخ ربات
    st.session_state.message_history.append({
        "sender": "assistant", 
        "text": bot_response_text,
        "time": current_time,
        "display_name": "AI Assistant"
    })