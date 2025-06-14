from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from streamlit_mic_recorder import mic_recorder, speech_to_text
from gtts.lang import tts_langs
import streamlit as st
from gtts import gTTS
import os
import time
import re

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import docx
from langchain.vectorstores import FAISS

# Initialize all session state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None    
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = []
if "language" not in st.session_state:
    st.session_state.language = "English"
if "book_loaded" not in st.session_state:
    st.session_state.book_loaded = False
if "current_book" not in st.session_state:
    st.session_state.current_book = ""
if "voice_input" not in st.session_state:
    st.session_state.voice_input = ""
if "recording" not in st.session_state:
    st.session_state.recording = False

# Enhanced bilingual story content
BILINGUAL_BOOK_CONTENT = {
    "English": {
        "title": "The Rabbit and The Tortoise",
        "content": """Once upon a time, in a forest, there lived a very fast rabbit. He was very proud of his speed and would often boast about it to other animals.

One day, he saw a tortoise walking slowly. The rabbit laughed and said, "You're so slow! I could beat you in a race without even trying."

The tortoise replied, "Let's have a race then."

All the animals of the forest gathered to watch. The fox signaled the start of the race. The rabbit ran very fast and was soon far ahead. On the way, he saw a tree.

"The tortoise is so far behind. I can take a short nap," thought the rabbit and slept under the tree.

Meanwhile, the tortoise kept walking slowly but steadily. While the rabbit was sleeping, the tortoise passed him and finally reached the finish line.

When the rabbit woke up, he saw that the tortoise had already won. All the animals were congratulating the tortoise.

The rabbit realized his mistake. That day he learned that "Slow and steady wins the race."

Moral: Overconfidence can lead to failure. Consistency and perseverance always win.""",
        "image": "https://media.istockphoto.com/id/899658028/vector/rabbit-and-tortoise-story.jpg?s=612x612&w=0&k=20&c=o9xLL98B4rqj0FPAr3X2iuDrE_3_m9WN2ozO2JCLgkM=",
        "questions": [
            "What is the moral of the story?",
            "What was the rabbit's mistake?",
            "How did the tortoise win?"
        ]
    },
    "Urdu": {
        "title": "خرگوش اور کچھوا",
        "content": """ایک جنگل میں ایک تیز رفتار خرگوش رہتا تھا۔ وہ ہر کسی کو اپنی دوڑ کی صلاحیت پر فخر کرتا تھا۔ ایک دن، اس نے ایک کچھوے کو دیکھا جو بہت آہستہ چل رہا تھا۔

خرگوش نے کچھوے سے کہا، "تم بہت سست ہو! میں تمہارے ساتھ دوڑ میں شاید ہی ہاروں۔"

کچھوے نے مسکراتے ہوئے جواب دیا، "چلو دوڑ لگاتے ہیں۔"

جنگل کے تمام جانور جمع ہوئے۔ لومڑی نے دوڑ کا اشارہ کیا۔ خرگوش تیز دوڑا اور بہت آگے نکل گیا۔ راستے میں اسے ایک درخت نظر آیا۔

"کچھوا تو بہت پیچھے ہے۔ میں تھوڑا سا آرام کر لوں،" خرگوش نے سوچا اور درخت کے نیچے سو گیا۔

دراں اثنا، کچھوا مسلسل آہستہ آہستہ چلتا رہا۔ جب خرگوش سویا ہوا تھا، کچھوا اس کے پاس سے گزر گیا اور آخرکار دوڑ کے اختتام تک پہنچ گیا۔

جب خرگوش جاگا تو دیکھا کہ کچھوا فتح کی لکیر پار کر چکا ہے۔ تمام جانور کچھوے کو مبارکباد دے رہے تھے۔

خرگوش کو اپنی غلطی کا احساس ہوا۔ اس دن اس نے سیکھا کہ "آہستہ اور مستقل رہنا تیز لیکن بے قاعدہ ہونے سے بہتر ہے۔"

اخلاقی سبق: تکبر کبھی فائدہ مند نہیں ہوتا۔ مستقل محنت ہمیشہ کامیابی لاتی ہے۔""",
        "image": "https://media.istockphoto.com/id/899658028/vector/rabbit-and-tortoise-story.jpg?s=612x612&w=0&k=20&c=o9xLL98B4rqj0FPAr3X2iuDrE_3_m9WN2ozO2JCLgkM=",
        "questions": [
            "کہانی کا اخلاقی سبق کیا ہے؟",
            "خرگوش کی کیا غلطی تھی؟",
            "کچھوا کس طرح جیت گیا؟"
        ]
    }
}

def toggle_language():
    st.session_state.language = "Urdu" if st.session_state.language == "English" else "English"
    st.session_state.book_loaded = False  # Reset book loaded state when language changes

def validate_urdu_answer(answer):
    """Simple validation to reduce hallucinations"""
    if not re.search(r'[\u0600-\u06FF]', answer) and st.session_state.language == "Urdu":
        return False
    if any(phrase in answer for phrase in ["نہیں معلوم", "پتہ نہیں", "کتاب میں نہیں"]):
        return False
    return True

# Dynamic CSS with font changes
css = '''
<style>
    /* Base styles */
    body {
        background-color: #000000 !important;
    }
    .stApp {
        background-color: #000000 !important;
    }
    .main {
        background-color: #000000 !important;
    }
    section.main {
        background-color: #000000 !important;
    }
    
    /* Neon-themed chat bubbles */
    .chat-message {
        padding: 1rem; 
        border-radius: 1.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        font-size: 1.1rem;
        border: 1px solid;
    }
    .chat-message.user {
        background-color: #000000;
        border-color: #00ff00;
        box-shadow: 0 0 10px #00ff00;
    }
    .chat-message.bot {
        background-color: #000000;
        border-color: #00bfff;
        box-shadow: 0 0 10px #00bfff;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-height: 80px !important;
        max-width: 80px !important;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #ffffff;
    }
    
    /* Neon buttons */
    button {
        font-size: 1.2rem !important;
        padding: 0.8rem 1.5rem !important;
        border-radius: 2rem !important;
        background-color: #000000 !important;
        color: #00ff00 !important;
        border: 2px solid #00ff00 !important;
        transition: all 0.3s ease !important;
    }
    button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 15px #00ff00 !important;
    }
    
    /* Special styling for language buttons */
    .toggle-btn .urdu-btn {
        background-color: #000000 !important;
        color: #ff00ff !important;
        border-color: #ff00ff !important;
    }
    .toggle-btn .eng-btn {
        background-color: #000000 !important;
        color: #00bfff !important;
        border-color: #00bfff !important;
    }
    
    /* Neon speak button */
    .speak-button {
        background: linear-gradient(145deg, #000000, #000000) !important;
        color: #ff00ff !important;
        font-size: 1.4rem !important;
        border: 2px solid #ff00ff !important;
        border-radius: 2rem !important;
        padding: 1rem 2rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 0 10px #ff00ff;
        transition: all 0.3s ease !important;
    }
    .speak-button:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 0 20px #ff00ff !important;
    }
    
    /* Loading animation */
    @keyframes neonPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .loading-animation {
        animation: neonPulse 1s infinite;
        font-size: 3rem;
        text-align: center;
        color: #00ff00;
    }
    
    /* File uploader with neon border */
    .stFileUploader>div>div {
        border: 2px dashed #00bfff !important;
        border-radius: 1rem !important;
        padding: 2rem !important;
        background-color: rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Text input styling */
    .stTextInput>div>div>input {
        color: white !important;
        background-color: #000000 !important;
        border: 1px solid #00ff00 !important;
    }
    
    /* Select box styling */
    .stSelectbox>div>div>div {
        color: white !important;
        background-color: #000000 !important;
        border: 1px solid #00bfff !important;
    }
    
    /* Urdu text specific styling */
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Urdu Typesetting', sans-serif !important;
        font-size: 1.2rem;
        line-height: 2;
        direction: rtl;
        text-align: right;
    }
    
    /* English text specific styling */
    .english-text {
        font-family: 'Arial', sans-serif !important;
    }
    
    /* Microphone button styling */
    .mic-button {
        background: linear-gradient(145deg, #000000, #000000) !important;
        color: #ff00ff !important;
        border: 2px solid #ff00ff !important;
        border-radius: 50% !important;
        width: 50px !important;
        height: 50px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin-left: 10px !important;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" style="max-height: 80px; max-width: 80px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message {{LANG_CLASS}}">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/1864/1864593.png" style="max-height: 80px; max-width: 80px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message {{LANG_CLASS}}">{{MSG}}</div>
</div>
'''

def display_conversation_and_audio():
    if 'conversation_history' in st.session_state:
        for i, message in enumerate(st.session_state.conversation_history):
            if isinstance(message, dict):
                lang_class = "urdu-text" if st.session_state.language == "Urdu" else "english-text"
                if message["role"] == "user":
                    st.markdown(
                        user_template.replace("{{MSG}}", message["content"])
                                      .replace("{{LANG_CLASS}}", lang_class), 
                        unsafe_allow_html=True
                    )
                elif message["role"] == "bot":
                    st.markdown(
                        bot_template.replace("{{MSG}}", message["content"])
                                    .replace("{{LANG_CLASS}}", lang_class), 
                        unsafe_allow_html=True
                    )
                    response_audio_file = f"response_audio_{(i//2)+1}.mp3"
                    if os.path.exists(response_audio_file):
                        st.audio(response_audio_file)

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        _, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += "Unsupported file type."
    return text

def get_pdf_text(pdf):
    try:
        reader = PdfReader(pdf)
        return "".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def get_docx_text(doc_file):
    try:
        doc = docx.Document(doc_file)
        return ' '.join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=900, chunk_overlap=100)
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

def get_conversation_chain(vectorstore, api_key=None, use_ollama=False):
    if use_ollama:
        llm = Ollama(model="tinyllama", temperature=0.3)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def get_voice_input():
    """Enhanced microphone input with language support"""
    if st.session_state.language == "Urdu":
        voice_text = speech_to_text(
            language='ur-PK',
            start_prompt="🎤 اردو میں بات کریں",
            stop_prompt="🛑 رک جائیں",
            use_container_width=False,
            just_once=True,
            key='urdu_voice_input'
        )
    else:
        voice_text = speech_to_text(
            language='en-US',
            start_prompt="🎤 Speak now",
            stop_prompt="🛑 Stop",
            use_container_width=False,
            just_once=True,
            key='english_voice_input'
        )
    
    if voice_text:
        st.session_state.voice_input = voice_text
        return voice_text
    return ""

def voice_input_component():
    """Complete voice input UI component"""
    input_col, mic_col = st.columns([5, 1])
    
    with input_col:
        prompt_placeholder = ("Type your message here..." if st.session_state.language == "English" 
                            else "اپنا پیغام یہاں ٹائپ کریں...")
        user_input = st.text_area(
            "",
            value=st.session_state.voice_input,
            placeholder=prompt_placeholder,
            key="text_input",
            height=100,
            label_visibility="collapsed"
        )
    
    with mic_col:
        if mic_recorder(
            key="mic_recorder",
            start_prompt="🎤",
            stop_prompt="🛑",
            just_once=True,
            format="webm"
        ):
            st.session_state.recording = True
            
        if st.session_state.recording:
            with st.spinner("Listening..." if st.session_state.language == "English" else "سن رہا ہوں..."):
                voice_text = get_voice_input()
                if voice_text:
                    st.session_state.voice_input = voice_text
                    st.session_state.recording = False
                    st.rerun()
    
    return user_input if user_input else st.session_state.voice_input

def handle_user_input(user_question):
    with st.spinner("Thinking..." if st.session_state.language == "English" else "سوچ رہا ہوں..."):
        try:
            response = st.session_state.conversation({'question': user_question})
            
            # Validate response
            if st.session_state.language == "Urdu" and not validate_urdu_answer(response['answer']):
                response['answer'] = "میں اس سوال کا صحیح جواب نہیں دے سکتا۔ براہ کرم کوئی اور سوال پوچھیں۔"
            
            st.session_state.conversation_history.append({"role": "user", "content": user_question})
            st.session_state.conversation_history.append({"role": "bot", "content": response['answer']})
            
            # Convert response to speech
            response_lang = "ur" if st.session_state.language == "Urdu" else "en"
            response_audio_file = f"response_audio_{len(st.session_state.audio_files)+1}.mp3"
            tts = gTTS(text=response['answer'], lang=response_lang)
            tts.save(response_audio_file)
            st.session_state.audio_files.append(response_audio_file)
            
            # Clear voice input after processing
            st.session_state.voice_input = ""
            st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")

def main():
    try:
        api_key = st.secrets["google_ai"]["api_key"]
    except:
        api_key = None
        st.warning("Google AI API key not found. Using Ollama as fallback.")

    # Header with neon styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://via.placeholder.com/300x100.png?text=Urdu+Learning+Buddy", width=300)
        
        if st.session_state.language == "Urdu":
            st.markdown("""
            <h1 style='text-align: center; color: #00ff00; font-family: "Jameel Noori Nastaleeq"; text-shadow: 0 0 10px #00ff00;'>
                اردو سیکھنے کا دوست
            </h1>
            <div style="text-align: center; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem; color: #00bfff; text-shadow: 0 0 5px #00bfff; font-family: 'Jameel Noori Nastaleeq';">آئیے مل کر سیکھیں!</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <h1 style='text-align: center; color: #00ff00; font-family: "Arial", sans-serif; text-shadow: 0 0 10px #00ff00;'>
                <span style="color: #00ff00;">U</span>
                <span style="color: #00bfff;">r</span>
                <span style="color: #ff00ff;">d</span>
                <span style="color: #00ff00;">u</span>
                Learning Buddy
            </h1>
            <div style="text-align: center; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem; color: #00bfff; text-shadow: 0 0 5px #00bfff;">Let's learn together!</span>
            </div>
            """, unsafe_allow_html=True)

    # Settings
    st.markdown("---")
    st.subheader("⚙️ Settings" if st.session_state.language == "English" else "⚙️ ترتیبات")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌍 Language" if st.session_state.language == "English" else "### 🌍 زبان")
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.button("English", key="eng_btn", on_click=toggle_language)
        with btn_col2:
            st.button("اردو", key="urdu_btn", on_click=toggle_language)
    
    with col2:
        option = st.selectbox(
            "📋 Choose an option" if st.session_state.language == "English" else "📋 اختیار منتخب کریں",
            ["Questions/Answers", "Children's Book"] if st.session_state.language == "English" 
            else ["عام سوالات", "بچوں کی کتاب"]
        )
    
    if st.button("🧹 Clear Chat", key="clear_chat"):
        st.session_state.conversation_history = []
        st.session_state.audio_files = []
        st.session_state.chat_history = []
        st.session_state.generated = []
        st.session_state.past = []
        st.session_state.entered_prompt = ""
        st.session_state.book_loaded = False
        st.session_state.voice_input = ""
        st.success("Chat history cleared!" if st.session_state.language == "English" else "بات چیت کی تاریخ صاف ہو گئی!")
        st.balloons()

    # Input section with microphone
    st.markdown("---")
    st.subheader("💬 Ask a question" if st.session_state.language == "English" else "💬 سوال پوچھیں")
    
    user_input = voice_input_component()
    
    submit_text = "Submit" if st.session_state.language == "English" else "جمع کرائیں"
    if st.button(submit_text, use_container_width=True) and user_input:
        with st.spinner("Processing..." if st.session_state.language == "English" else "پروسیس ہو رہا ہے..."):
            try:
                # Set up language templates
                if st.session_state.language == "Urdu":
                    chat_template = ChatPromptTemplate.from_messages([
                        ("system", "آپ کا اردو سیکھنے میں مددگار دوست ہوں۔ آسان الفاظ اور مختصر جملے استعمال کریں۔"),
                        ("human", "{human_input}"),
                    ])
                    response_lang = "ur"
                else:
                    chat_template = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful AI assistant that teaches English to children. Use simple words and short sentences."),
                        ("human", "{human_input}"),
                    ])
                    response_lang = "en"

                if api_key:
                    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                    chain = chat_template | model | StrOutputParser()
                else:
                    model = Ollama(model="tinyllama")
                    chain = chat_template | model | StrOutputParser()

                response = chain.invoke({"human_input": user_input})
                
                # Validate response
                if st.session_state.language == "Urdu" and not validate_urdu_answer(response):
                    response = "میں اس سوال کا صحیح جواب نہیں دے سکتا۔ براہ کرم کوئی اور سوال پوچھیں۔"
                
                st.session_state.conversation_history.append({"role": "user", "content": user_input})
                st.session_state.conversation_history.append({"role": "bot", "content": response})
                
                # Convert response to speech
                response_audio_file = f"response_audio_{len(st.session_state.audio_files)+1}.mp3"
                tts = gTTS(text=response, lang=response_lang)
                tts.save(response_audio_file)
                st.session_state.audio_files.append(response_audio_file)
                
                # Clear voice input after processing
                st.session_state.voice_input = ""
            except Exception as e:
                st.error(f"Error: {str(e)}")

    display_conversation_and_audio()

    if option == "Children's Book" or option == "بچوں کی کتاب":
        book_data = BILINGUAL_BOOK_CONTENT[st.session_state.language]
        
        st.subheader(f"📖 {book_data['title']}")
        
        # Automatically load the story
        if not st.session_state.book_loaded:
            with st.spinner("Preparing the story..." if st.session_state.language == "English" else "کہانی تیار کی جا رہی ہے..."):
                text_chunks = get_text_chunks(book_data["content"])
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore, 
                    api_key=api_key,
                    use_ollama=True if not api_key else False
                )
                st.session_state.book_loaded = True
        
        # Display the story content with proper formatting
        lang_class = "urdu-text" if st.session_state.language == "Urdu" else "english-text"
        st.markdown(f"""
        <div class="{lang_class}" style="background-color: rgba(0,0,0,0.7); padding: 20px; border-radius: 10px; border: 1px solid #00ff00;">
            {book_data["content"]}
        </div>
        """, unsafe_allow_html=True)
        
        st.image(book_data["image"], width=300)
        
        # Sample questions
        st.subheader("Sample Questions" if st.session_state.language == "English" else "نمونہ سوالات")
        cols = st.columns(3)
        
        for i, question in enumerate(book_data["questions"]):
            if cols[i%3].button(question, use_container_width=True):
                st.session_state.voice_input = question
                st.rerun()

    elif option == "Document Q/A" or option == "دستاویزات سے سوالات":
        st.subheader("📚 Document Question and Answer" if st.session_state.language == "English" else "📚 دستاویزات سے سوالات")
        uploaded_files = st.file_uploader(
            "Upload your files" if st.session_state.language == "English" else "اپنی فائلیں اپ لوڈ کریں",
            type=["pdf", "docx"], 
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Process Documents" if st.session_state.language == "English" else "دستاویزات پروسیس کریں"):
            with st.spinner("Processing documents..." if st.session_state.language == "English" else "دستاویزات پروسیس ہو رہی ہیں..."):
                try:
                    raw_text = get_files_text(uploaded_files)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
                    st.session_state.processComplete = True
                    st.success("Documents processed successfully!" if st.session_state.language == "English" else "دستاویزات کامیابی سے پروسیس ہو گئیں!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

        if st.session_state.processComplete:
            if st.session_state.language == "Urdu":
                user_question = get_voice_input()
                if user_question:
                    st.markdown(f"""
                    <div style="background-color: rgba(0,0,0,0.5); padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #ff00ff;">
                        <h4 style="color: #ff00ff; margin-bottom: 5px;">آپ کا سوال:</h4>
                        <p style="font-size: 16px; color: white;">{user_question}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                user_question = get_voice_input()
                if user_question:
                    st.markdown(f"""
                    <div style="background-color: rgba(0,0,0,0.5); padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #00bfff;">
                        <h4 style="color: #00bfff; margin-bottom: 5px;">Your Question:</h4>
                        <p style="font-size: 16px; color: white;">{user_question}</p>
                    </div>
                    """, unsafe_allow_html=True)

            if user_question:
                handle_user_input(user_question)

if __name__ == "__main__":
    main()