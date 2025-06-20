import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import numpy as np
import pickle
import json
import random
import ast
import torch
from tensorflow.keras.models import load_model

# Flexible transformer imports
try:
    from transformers import AutoTokenizer, AutoModel
    Tokenizer, Model = AutoTokenizer, AutoModel
except ImportError:
    try:
        from transformers import BertTokenizer, BertModel
        Tokenizer, Model = BertTokenizer, BertModel
    except ImportError:
        st.error("Please install transformers: pip install transformers")
        st.stop()

# App setup
st.set_page_config(page_title="First Aid Chatbot", page_icon="🩹")
st.title("🩹 First Aid Chatbot")

@st.cache_resource(show_spinner="Loading first aid knowledge...")
def load_resources():
    try:
        model = load_model('models/chatbot_model.h5')
        
        with open('models/words.pkl', 'rb') as f:
            words = pickle.load(f)
        
        with open('models/classes.pkl', 'rb') as f:
            classes = pickle.load(f)
        
        with open("data/chat/chatbot_intents.json") as f:
            raw_data = json.load(f)
        
        # Properly parse the stringified lists
        first_aid_data = {"intents": []}
        for intent in raw_data["intents"]:
            try:
                patterns = ast.literal_eval(intent["patterns"][0]) if isinstance(intent["patterns"], list) else []
                responses = ast.literal_eval(intent["responses"][0]) if isinstance(intent["responses"], list) else []
            except:
                patterns = []
                responses = []
            
            first_aid_data["intents"].append({
                "tags": intent["tags"],
                "patterns": patterns,
                "responses": responses
            })
        
        tokenizer = Tokenizer.from_pretrained('bert-base-uncased')
        bert_model = Model.from_pretrained('bert-base-uncased')
        
        return model, words, classes, first_aid_data, tokenizer, bert_model
        
    except Exception as e:
        st.error(f"Loading error: {str(e)}")
        st.stop()

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I can help with:\n• Cuts\n• Abrasions\n• Stings\n• Splinters\n• Sprains\n• Strains\n\nWhat do you need help with?"
    }]

# Load resources
try:
    model, words, classes, first_aid_data, tokenizer, bert_model = load_resources()
    st.success("Ready to help with first aid advice!")
except Exception as e:
    st.error(f"Startup failed: {str(e)}")
    st.stop()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def predict_intent(text):
    embedding = get_embedding(text)
    prediction = model.predict(embedding)[0]
    return classes[np.argmax(prediction)]

def format_response(response):
    # Clean up the response formatting
    response = response.replace("['", "").replace("']", "").replace("\\n", "\n")
    if '\n' in response:
        steps = [step.strip() for step in response.split('\n') if step.strip()]
        return "First aid steps:\n\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
    return response

def get_first_aid_response(intent):
    intent = intent.lower()
    
    # First try exact tag matching
    for category in first_aid_data['intents']:
        if any(tag.lower() == intent for tag in category['tags']):
            if category['responses']:
                return format_response(random.choice(category['responses']))
    
    # Then try keyword matching
    keyword_mapping = {
        'cut': 'cuts',
        'abrasion': 'abrasions',
        'sting': 'stings',
        'splinter': 'splinter',
        'sprain': 'sprains',
        'strain': 'strains'
    }
    
    for keyword, tag in keyword_mapping.items():
        if keyword in intent:
            for category in first_aid_data['intents']:
                if tag in [t.lower() for t in category['tags']]:
                    if category['responses']:
                        return format_response(random.choice(category['responses']))
    
    return "I understand you need first aid help. Could you specify if it's for:\n- Cuts\n- Abrasions\n- Stings\n- Splinters\n- Sprains\n- Strains"

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process input
if prompt := st.chat_input("What first aid help do you need?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                intent = predict_intent(prompt)
                response = get_first_aid_response(intent)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                fallback = get_first_aid_response(prompt.lower())
                st.markdown(fallback)
                st.session_state.messages.append({"role": "assistant", "content": fallback})

# Sidebar with help
with st.sidebar:
    st.header("🚑 Emergency Contacts")
    st.markdown("""
    - **Local Emergency**: 911 (KGL)
    - **Poison Control**: 0792402851 (RW)
    """)
    
    st.header("💡 Quick First Aid Tips")
    st.markdown("""
    - **Cuts**: Clean, pressure, bandage
    - **Abrasions**: Clean gently, antibiotic ointment
    - **Stings**: Remove stinger, ice, antihistamine
    - **Splinters**: Epsom salts soak or vinegar
    - **Sprains**: RICE method (Rest, Ice, Compression, Elevation)
    - **Strains**: Rest, ice, gentle stretching
    """)
    st.warning("For serious emergencies,Please seek professional medical help immediately.")