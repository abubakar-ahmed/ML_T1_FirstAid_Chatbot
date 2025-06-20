---
###   https://first-aid-chatbot.streamlit.app/
# 🩹 First Aid Assistant Chatbot

## A Streamlit-based chatbot that provides immediate first aid guidance for common medical situations using BERT embeddings and deep learning classification.

---

## 📚 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contact](#contact)

---

## ✨ Features

- **Interactive Chat Interface** – Natural conversation flow
- **Comprehensive First Aid Coverage**:
  - Cuts & wounds
  - Burns
  - Sprains & strains
  - Choking
  - Nosebleeds
  - Stings
- **Emergency Quick Reference** – Sidebar with essential contacts and tips  
- **Responsive Design** – Works on mobile and desktop

---

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/abubakar-ahmed/ML_T1_FirstAid_Chatbot
cd first-aid-chatbot
````

### 2. Create and Activate Virtual Environment

```bash
# For Linux/macOS
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Application
### Access the Chatbot

* The app will open in your default browser at:

  ```
  https://first-aid-chatbot.streamlit.app/
  ```
* Or visit the URL shown in your terminal.

### Interact with the Chatbot

Type your first aid question in the chat input.
Example queries:

-- Cuts
-- Sprain
-- Splinter
-- Nasal Congestions
-- Vertigo

---

## 📁 Project Structure

```
first-aid-chatbot/
├── app.py                        # Main Streamlit app
├── models/
│   ├── chatbot_model.h5          # Trained Keras model
│   ├── words.pkl                 # Vocabulary pickle
│   └── classes.pkl               # Intent class labels
├── data/
│   └── chat/
│       └── chatbot_intents.json  # First aid knowledge base
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

##  Technical Details

### Architecture

#### Natural Language Understanding:

* BERT embeddings (via HuggingFace Transformers)
* Custom tokenizer for user inputs

#### Intent Classification:

* TensorFlow/Keras deep learning model
* Trained on labeled first aid conversations

#### Response Generation:

* Pulls relevant response from `chatbot_intents.json`
* Formats and presents step-by-step guidance

### Dependencies

| Package      | Version | Purpose                    |
| ------------ | ------- | -------------------------- |
| streamlit    | >=1.0   | Web UI                     |
| tensorflow   | >=2.0   | Model inference            |
| transformers | >=4.0   | BERT embeddings            |
| torch        | >=1.0   | PyTorch backend (optional) |
| nltk         | >=3.0   | Text pre-processing        |
---

## Contact

* **Author:** Abubakar Ahmed 
* **GitHub:** [@abubakar-ahmed](https://github.com/abubakar-ahmed)
* **Email:** [a.ahmed1@alustudent.com]
