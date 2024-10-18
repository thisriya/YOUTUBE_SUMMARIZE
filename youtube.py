# #WORKS ONLY FOR THE ENGLISH VIDEOS
#  import streamlit as st
#  from dotenv import load_dotenv
#  load_dotenv()
#  from youtube_transcript_api import YouTubeTranscriptApi
#  import google.generativeai as genai
#  import os
#  genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#  prompt="you are  autube video summazrizer"

#  def extract_transcript_details(youtube_video_url):
#      try:
#          video_id=youtube_video_url.split("=")[1]
#          print(video_id)
#          transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
#          transcript=""
#          for i in transcript_text:
#              transcript+= " " +i["text"]
#          return transcript
#      except Exception as e:
#          raise e

#  def generate_gemini_content(transcript_text,prompt):
#      model=genai.GenerativeModel("gemini-pro")
#      response=model.generate_content(prompt+transcript_text)
#      return response.text

#  st.title("Youtube Trancript to detailed Notes Cnverter")
#  youtube_link=st.text_input("Enter youtube link:")
#  if youtube_link:
#      video_id=youtube_link.split("=")[1]
#      print(video_id)
#      st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg",use_column_width=True)
#  if st.button("Get Detailed Notes"):
#      transcript_text=extract_transcript_details(youtube_link)

#      if transcript_text:
#          summary=generate_gemini_content(transcript_text,prompt)
#          st.markdown("## Detailed Notes")
#          st.write(summary)


#WORKS FOR VIDEOS HAVING DECENT HINDI
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import os
from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
import google.generativeai as genai
import time
import logging
import re

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load environment variables
load_dotenv()

# Configure Google Gemini API
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Google API Key is missing. Please set it in the .env file.")
    st.stop()

genai.configure(api_key=google_api_key)
prompt = "Summarize the following YouTube video transcript:"

# Function to extract transcript from YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        # Extract video ID more robustly
        if "v=" in youtube_video_url:
            video_id = youtube_video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_video_url:
            video_id = youtube_video_url.split("youtu.be/")[1].split("?")[0]
        else:
            st.error("Invalid YouTube URL format.")
            logging.error(f"Invalid YouTube URL format: {youtube_video_url}")
            return None

        # Fetch transcript in both English and Hindi
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
        # Combine transcript text
        transcript = " ".join([item["text"] for item in transcript_data])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        logging.error(f"Error fetching transcript for URL {youtube_video_url}: {e}")
        return None

# Function to split text into manageable chunks
def split_text(text, max_length=500):
    """
    Splits the text into smaller chunks to avoid exceeding model limits.
    """
    words = text.split()
    chunks = []
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = word
        else:
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Function to load and cache translation model
@st.cache_resource
def load_translation_model(source_lang: str, target_lang: str):
    """
    Load and cache the appropriate MarianMT model and tokenizer based on source and target languages.
    """
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading translation model: {e}")
        logging.error(f"Error loading translation model {model_name}: {e}")
        return None, None

# Function to translate text to English using Hugging Face Transformers
def translate_to_english(text, tokenizer, model, batch_size=8):
    """
    Translates the given text to English using the provided tokenizer and model.
    Handles large texts by splitting into manageable chunks and translating in batches.
    """
    try:
        translated_chunks = []
        chunks = split_text(text, max_length=500)  # Adjust max_length as needed
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        progress_bar = st.progress(0)
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            inputs = tokenizer(batch_chunks, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                translated = model.generate(**inputs)
            translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
            translated_chunks.extend(translated_texts)
            progress = min((i + batch_size) / len(chunks), 1.0)
            progress_bar.progress(progress)
            st.write(f"Translated batch {i//batch_size +1} of {total_batches}")
        return " ".join(translated_chunks)
    except Exception as e:
        st.error(f"An unexpected error occurred during translation: {e}")
        logging.error(f"Unexpected error during translation: {e}")
        return None

# Function to generate summary using Gemini API
def generate_gemini_content(transcript_text, prompt, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt + transcript_text)
            
            # Check if the response contains a valid text part
            if response and hasattr(response, 'text') and response.text:
                return response.text
            else:
                st.warning(f"Attempt {attempt}: Summary was blocked by safety filters.")
                logging.warning(f"Attempt {attempt}: Summary was blocked by safety filters.")
                time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            st.warning(f"Attempt {attempt}: Error generating summary: {e}")
            logging.warning(f"Attempt {attempt}: Error generating summary: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    st.error("Failed to generate summary after multiple attempts.")
    return None

# Function to generate summary using Hugging Face's summarization model (alternative)
@st.cache_resource
def load_summarization_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def generate_summary_huggingface(transcript_text, max_length=150, min_length=30):
    try:
        summarizer = load_summarization_model()
        summary = summarizer(transcript_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        logging.error(f"Error generating summary: {e}")
        return None

# Streamlit app interface
st.title("YouTube Transcript to Detailed Notes Converter")
youtube_link = st.text_input("Enter YouTube link:")

if youtube_link:
    try:
        # Extract video ID to display thumbnail
        if "v=" in youtube_link:
            video_id = youtube_link.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_link:
            video_id = youtube_link.split("youtu.be/")[1].split("?")[0]
        else:
            st.error("Invalid YouTube URL format.")
            video_id = None

        if video_id:
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    except IndexError:
        st.error("Invalid YouTube URL. Please enter a correct link.")

if st.button("Get Detailed Notes"):
    if not youtube_link:
        st.error("Please enter a YouTube link.")
    else:
        with st.spinner("Extracting transcript..."):
            transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            # Determine if the transcript is in Hindi or English
            # Simple heuristic: check if most characters are in Devanagari script
            devanagari_pattern = re.compile(r'[\u0900-\u097F]')
            devanagari_chars = re.findall(devanagari_pattern, transcript_text)
            is_hindi = len(devanagari_chars) > len(transcript_text) / 2  # Arbitrary threshold

            if is_hindi:
                source_lang = 'hi'
                target_lang = 'en'
                st.info("Detected language: Hindi. Translating to English...")
            else:
                source_lang = 'en'
                target_lang = 'hi'
                st.info("Detected language: English. No translation needed.")

            if is_hindi:
                with st.spinner("Loading translation model..."):
                    tokenizer, model = load_translation_model(source_lang, target_lang)

                if tokenizer and model:
                    with st.spinner("Translating transcript to English..."):
                        english_transcript = translate_to_english(transcript_text, tokenizer, model)
            else:
                english_transcript = transcript_text

            if english_transcript:
                # Choose between Gemini API and Hugging Face summarization
                st.info("Generating summary...")
                summary = generate_gemini_content(english_transcript, prompt)
                
                if not summary:
                    st.warning("Falling back to alternative summarization method...")
                    summary = generate_summary_huggingface(english_transcript)

                if summary:
                    st.markdown("## Detailed Notes")
                    st.write(summary)
