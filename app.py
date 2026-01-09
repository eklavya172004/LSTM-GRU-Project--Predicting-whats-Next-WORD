import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page config
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Dark Theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Dark Theme */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    html, body, [data-testid="stAppViewContainer"], .main {
        background: #0a0e27 !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at top, #1a1f3a 0%, #0a0e27 50%, #000000 100%) !important;
    }
    
    /* Animated gradient background */
    .main {
        background: #0a0e27 !important;
        position: relative;
    }
    
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(139, 92, 246, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 3rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #a0aec0;
        font-weight: 300;
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.8s ease-out 0.2s both;
    }
    
    .hero-description {
        font-size: 1rem;
        color: #718096;
        font-weight: 400;
        animation: fadeInUp 0.8s ease-out 0.4s both;
    }
    
    /* Glassmorphism Card */
    .glass-card {
        background: rgba(26, 32, 58, 0.6);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
        box-shadow: 
            0 8px 32px 0 rgba(0, 0, 0, 0.4),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        position: relative;
        z-index: 1;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: 
            0 12px 48px 0 rgba(139, 92, 246, 0.15),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        transform: translateY(-2px);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-header::before {
        content: '';
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    
    /* Input Styling */
    .stTextInput input {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 2px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 16px !important;
        color: #e2e8f0 !important;
        font-size: 1.1rem !important;
        padding: 16px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: rgba(139, 92, 246, 0.8) !important;
        box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.1) !important;
        background: rgba(15, 23, 42, 0.95) !important;
    }
    
    .stTextInput input::placeholder {
        color: #64748b !important;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 14px 32px !important;
        border-radius: 12px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stButton button:active {
        transform: translateY(0px) !important;
    }
    
    /* Example Buttons */
    .stButton button[kind="secondary"] {
        background: rgba(51, 65, 85, 0.6) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        color: #cbd5e1 !important;
        box-shadow: none !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        background: rgba(71, 85, 105, 0.8) !important;
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.2) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    .stMetric {
        background: rgba(30, 41, 59, 0.5) !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(100, 116, 139, 0.2) !important;
    }
    
    /* Alert Boxes */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 12px !important;
        color: #6ee7b7 !important;
        padding: 1rem !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 12px !important;
        color: #93c5fd !important;
        padding: 1rem !important;
    }
    
    .stWarning {
        background: rgba(251, 191, 36, 0.1) !important;
        border: 1px solid rgba(251, 191, 36, 0.3) !important;
        border-radius: 12px !important;
        color: #fcd34d !important;
        padding: 1rem !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 12px !important;
        color: #fca5a5 !important;
        padding: 1rem !important;
    }
    
    /* Text Colors */
    p, span, label, li {
        color: #cbd5e1 !important;
    }
    
    strong {
        color: #e2e8f0 !important;
    }
    
    code {
        background: rgba(139, 92, 246, 0.15) !important;
        color: #c4b5fd !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    
    /* Architecture List */
    .architecture-list {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 4rem;
        padding: 2rem;
        border-top: 1px solid rgba(100, 116, 139, 0.2);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Load the model (with caching)
@st.cache_resource
def load_resources():
    try:
        model = load_model('next_word_lstm.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer, None
    except FileNotFoundError as e:
        return None, None, str(e)

# Load resources
model, tokenizer, error = load_resources()

# Hero Section
st.markdown('''
    <div class="hero-container">
        <div class="hero-title">✨ Hamlet Oracle</div>
        <div class="hero-subtitle">AI-Powered Next Word Prediction</div>
        <div class="hero-description">Trained on Shakespeare's Hamlet using LSTM Neural Networks</div>
    </div>
''', unsafe_allow_html=True)

if model and tokenizer:
    # Main content layout
    col1, col2 = st.columns([1.8, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">🎯 Text Prediction</div>', unsafe_allow_html=True)
        
        # Input section
        user_input = st.text_input(
            "Enter your text",
            placeholder="Type words from Hamlet... e.g., 'To be or not'",
            help="Enter a sequence of words to predict the next word",
            label_visibility="collapsed"
        )
        
        # Prediction logic
        def predicted_word(model, tokenizer, text, max_sequence_len):
            token_list = tokenizer.texts_to_sequences([text])[0]
            
            if len(token_list) >= max_sequence_len:
                token_list = token_list[-(max_sequence_len - 1):]
            
            token_list = pad_sequences(
                [token_list],
                maxlen=max_sequence_len - 1,
                padding='pre'
            )
            
            predictions = model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predictions, axis=1)[0]
            
            return tokenizer.index_word.get(predicted_word_index, None)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Predict button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            predict_btn = st.button("🔮 Predict Next Word", use_container_width=True)
        
        if predict_btn:
            if user_input.strip():
                max_sequence_len = model.input_shape[1] + 1
                next_word = predicted_word(model, tokenizer, user_input, max_sequence_len)
                
                if next_word:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.success(f"**Predicted Word:** `{next_word}`")
                    st.info(f"**Complete Sequence:** {user_input} **{next_word}**")
                else:
                    st.warning("Unable to predict. Try different text or check model vocabulary.")
            else:
                st.warning("Please enter some text to get a prediction.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">💫 Quick Examples</div>', unsafe_allow_html=True)
        
        examples = [
            ("To be or not", "🎭"),
            ("Something is rotten", "💀"),
            ("What a piece", "🌟"),
            ("Barnardo. Who's", "👤"),
        ]
        
        for example, emoji in examples:
            if st.button(f'{emoji} {example}', use_container_width=True, key=example):
                max_sequence_len = model.input_shape[1] + 1
                next_word = predicted_word(model, tokenizer, example, max_sequence_len)
                if next_word:
                    st.success(f"**→** `{next_word}`")
                else:
                    st.warning("No prediction available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Metrics Section
    st.markdown('<br><br>', unsafe_allow_html=True)
    # st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Model Information</div>', unsafe_allow_html=True)
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    with col_metric1:
        st.metric("Training Dataset", "Shakespeare's Hamlet")
    
    with col_metric2:
        st.metric("Model Architecture", "LSTM Neural Network")
    
    with col_metric3:
        st.metric("Vocabulary Size", "~5000 words")
    
    st.markdown('<div class="architecture-list">', unsafe_allow_html=True)
    st.markdown("""
    **Neural Network Architecture:**
    - **Embedding Layer** — Converts words to dense vector representations
    - **LSTM Layer 1** — 150 units with dropout regularization
    - **LSTM Layer 2** — 100 units for deeper pattern learning
    - **Dense Output** — Softmax activation for word probability distribution
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="custom-footer">
        Crafted with passion using <strong>Streamlit</strong> & <strong>TensorFlow</strong><br>
        <span style="color: #667eea;">●</span> 
        <span style="color: #764ba2;">●</span> 
        <span style="color: #f093fb;">●</span>
    </div>
    """, unsafe_allow_html=True)

else:
    # st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.error(f"Failed to load model files. Error: {error}")
    st.info("**Required files:**\n- `next_word_lstm.h5` (trained model)\n- `tokenizer.pickle` (vocabulary tokenizer)")
    st.markdown('</div>', unsafe_allow_html=True)