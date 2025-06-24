import os
import tempfile
import streamlit as st
from deep_translator import GoogleTranslator
import re
from dotenv import load_dotenv
from groq import Groq
import torch
try:
    from transformers import BertForSequenceClassification, BertTokenizer
except ImportError:
    # Alternative import paths if the direct import fails
    from transformers.models.bert.modeling_bert import BertForSequenceClassification
    from transformers.models.bert.tokenization_bert import BertTokenizer
try:
    from langdetect import detect
except ImportError:
    detect = None

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Speech & Text Sentiment Analysis",
    page_icon="🎤",
    layout="centered"
)

# Get API key from environment variable
API_KEY = os.getenv("API_KEY") or os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL", "whisper-large-v3-turbo")

# Configure Groq client
if API_KEY:
    os.environ["GROQ_API_KEY"] = API_KEY

# ========== Load Sentiment Analysis Model ==========
@st.cache_resource
def load_sentiment_model():
    save_path = "."
    try:
        # First attempt: Standard loading approach
        loaded_tokenizer = BertTokenizer.from_pretrained(save_path)
        loaded_model = BertForSequenceClassification.from_pretrained(save_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model.to(device)
    except NotImplementedError as e:
        # Handle meta tensor issue by avoiding direct .to() call
        st.warning("Detected meta tensor, using alternative loading method...")
        loaded_tokenizer = BertTokenizer.from_pretrained(save_path)
        
        # Use 'device_map' parameter to handle device placement properly
        loaded_model = BertForSequenceClassification.from_pretrained(
            save_path, 
            device_map="auto" if torch.cuda.is_available() else None
        )
        # device already handled by device_map="auto"
        device = next(loaded_model.parameters()).device  # Get the actual device used
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Attempting to load from Hugging Face Hub")
        try:
            loaded_tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
            loaded_model = BertForSequenceClassification.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment",
                device_map="auto" if torch.cuda.is_available() else None
            )
            device = next(loaded_model.parameters()).device  # Get the actual device used
        except Exception as e:
            st.error(f"Failed to load fallback model: {str(e)}")
            loaded_model = None
            loaded_tokenizer = None
            device = None
            
    return loaded_model, loaded_tokenizer, device

# Load the model
model, tokenizer, device = load_sentiment_model()

# Check if text contains Arabic
def is_arabic(text):
    """Check if text contains Arabic"""
    if not text:
        return False
        
    # Check for Arabic Unicode character range
    has_arabic_chars = any('\u0600' <= c <= '\u06FF' for c in text)
    
    # Check for common Arabic words and transliterations
    arabic_patterns = [
        'salam', 'salaam', 'asalam', 'assalam', 'alaikum', 'alikum',
        'bismillah', 'inshallah', 'habibi', 'shukran',
        'ال', 'في', 'من', 'هو', 'هي', 'انا', 'انت', 'جميل',
        'السلام', 'عليكم', 'ورحمة', 'وبركاته', 'الله'
    ]
    
    pattern_match = any(pattern.lower() in text.lower() for pattern in arabic_patterns)
    
    return has_arabic_chars or pattern_match

def analyze_sentiment(text, is_arabic=False):
    """Analyze the sentiment of text using BERT model"""
    if model is None or tokenizer is None:
        st.error("Sentiment analysis model is not properly loaded. Cannot perform prediction.")
        return -1
    
    # Try to detect language automatically if langdetect is available
    detected_lang = None
    if detect is not None:
        try:
            detected_lang = detect(text)
            # Override the is_arabic parameter with detection result
            is_arabic = detected_lang == "ar"
        except:
            pass  # Fall back to the provided is_arabic parameter
        
    # If Arabic, translate to English first
    if is_arabic:
        try:
            translated = GoogleTranslator(source='ar', target='en').translate(text)
            st.info(f"🌍 Translated for sentiment analysis: {translated}")
            text = translated
        except Exception as e:
            st.error(f"Translation error during sentiment analysis: {str(e)}")
            return -1
    
    # Tokenize & Predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class

def transcribe_audio(audio_file_path):
    """Transcribe audio using Groq Whisper API"""
    try:
        # Initialize the Groq client
        client = Groq()
        
        # Open the audio file
        with open(audio_file_path, "rb") as file:
            audio_data = file.read()
        
        # Transcribe the audio using Groq API
        with st.spinner(f"Transcribing using Groq's {MODEL}..."):
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), audio_data),
                model=MODEL,
                response_format="verbose_json"
            )
        
        # Get transcription text
        text = transcription.text
        
        # Determine if the text is Arabic
        has_arabic = is_arabic(text)
        
        # Try to get the language from the response if available
        language = getattr(transcription, "language", None)
        
        # If language not available, make a best guess
        if not language:
            if has_arabic:
                language = "ar"
            else:
                language = "en"  # Default to English
        
        return {
            "text": text,
            "language": language,
            "success": True,
            "has_arabic": has_arabic,
            "full_response": transcription
        }
        
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return {
            "text": None,
            "language": None,
            "success": False,
            "error": str(e)
        }

def main():
    st.title("Speech & Text Sentiment Analysis")
    st.write("Analyze sentiment in text or speech with support for Arabic and English!")
    
    # Simple info section in sidebar
    st.sidebar.info(
        f"This app uses Groq's {MODEL} model for speech recognition "
        "and BERT for sentiment analysis, with excellent multilingual support."
    )
    
    # Create tabs
    tab1, tab2 = st.tabs(["✍️ Text Analysis", "🎙️ Voice Analysis"])
    
    # Tab 1: Text Input Analysis
    with tab1:
        st.header("Text Sentiment Analysis")
        st.write("Enter text to analyze its sentiment")
        
        # Text input
        user_input = st.text_area("Your Text", height=150)
        
        if st.button("Analyze Text"):
            if user_input.strip() == "":
                st.warning("Please enter some text!")
            else:
                # Show spinner while processing
                with st.spinner("Analyzing sentiment..."):
                    # Detect if text is Arabic
                    text_contains_arabic = is_arabic(user_input)
                    
                    # Display detected language
                    lang_name = "Arabic" if text_contains_arabic else "English"
                    st.info(f"Detected Language: {lang_name}")
                    
                    # If Arabic, show the translation
                    if text_contains_arabic:
                        try:
                            translation = GoogleTranslator(source='ar', target='en').translate(user_input)
                            st.subheader("Translation:")
                            st.write(translation)
                            
                            # If it's an Islamic greeting, show the traditional response
                            if any(greeting in user_input.lower() for greeting in 
                                ['salam', 'salaam', 'asalam', 'assalam', 'السلام']):
                                st.success("Traditional response: Walaikum Assalam (وعليكم السلام)")
                        except Exception as e:
                            st.error(f"Translation error: {str(e)}")
                    
                    # Analyze sentiment
                    st.subheader("Sentiment Analysis:")
                    sentiment = analyze_sentiment(user_input, is_arabic=text_contains_arabic)
                    
                    if sentiment == -1:
                        # Error already displayed by the sentiment function
                        pass
                    elif sentiment == 2:
                        st.success("✅ Positive sentiment detected!")
                    elif sentiment == 1:
                        st.info("😐 Neutral sentiment detected!")
                    else:
                        st.error("⚠️ Negative sentiment detected!")
    
    # Tab 2: Voice Analysis
    with tab2:
        st.header("Voice-to-Text Sentiment Analysis")
        st.write("Upload an audio file for transcription and sentiment analysis")
        
        # Upload audio file
        uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, M4A, etc.)", 
                                        type=["wav", "mp3", "m4a", "webm", "mp4", "mpeg", "mpga", "ogg"])
        
        if uploaded_file is not None:
            # Display uploaded audio
            st.audio(uploaded_file, format="audio/wav")
            
            if st.button("Transcribe and Analyze"):
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    audio_file_path = temp_file.name
                
                # Transcribe the audio
                with st.spinner("Transcribing and analyzing..."):
                    result = transcribe_audio(audio_file_path)
                
                # Clean up temp file
                os.unlink(audio_file_path)
                
                if result["success"]:
                    # Display transcribed text
                    st.subheader("Transcription Result:")
                    st.write(result["text"])
                    
                    # Display detected language
                    if result["language"]:
                        lang_name = "Arabic" if result["language"] == "ar" else "English" if result["language"] == "en" else result["language"]
                        st.info(f"Detected Language: {lang_name}")
                    
                    # Text for sentiment analysis
                    text_for_sentiment = result["text"]
                    text_contains_arabic = result["language"] == "ar" or result["has_arabic"]
                    
                    # If Arabic, show the translation
                    if text_contains_arabic:
                        st.subheader("Translation:")
                        try:
                            translation = GoogleTranslator(source='ar', target='en').translate(result["text"])
                            st.write(translation)
                            
                            # If it's an Islamic greeting, show the traditional response
                            if any(greeting in result["text"].lower() for greeting in 
                                ['salam', 'salaam', 'asalam', 'assalam', 'السلام']):
                                st.success("Traditional response: Walaikum Assalam (وعليكم السلام)")
                        except Exception as e:
                            st.error(f"Translation error: {str(e)}")
                    
                    # Analyze sentiment
                    st.subheader("Sentiment Analysis:")
                    sentiment = analyze_sentiment(text_for_sentiment, is_arabic=text_contains_arabic)
                    
                    if sentiment == -1:
                        # Error already displayed by the sentiment function
                        pass
                    elif sentiment == 2:
                        st.success("✅ Positive sentiment detected!")
                    elif sentiment == 1:
                        st.info("😐 Neutral sentiment detected!")
                    else:
                        st.error("⚠️ Negative sentiment detected!")
                else:
                    st.error("Transcription failed. Please try again.")
    
    st.markdown("---")
    st.caption(f"Powered by Groq's {MODEL} and BERT Sentiment Analysis")

if __name__ == "__main__":
    main()