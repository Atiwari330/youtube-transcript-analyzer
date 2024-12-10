import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from youtube_transcript_api.formatters import TextFormatter
import re
import google.generativeai as genai
import os
from dotenv import load_dotenv
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import requests
import pandas as pd
from datetime import datetime
import json
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- NBA Players Fetcher Class ---
class NBAPlayersFetcher:
    def __init__(self):
        self.base_url = "https://stats.nba.com/stats/commonallplayers"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true'
        }
        self.cache_file = 'nba_players_cache.json'
        self.cache_duration = 24 * 60 * 60  # 24 hours in seconds
        self.session = self._create_session_with_retry()

    def _create_session_with_retry(self, retries=5, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)):
        """Creates a requests session with retry logic."""
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _load_cache(self):
        """Load cached data if it exists and is not expired."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    if time.time() - cache['timestamp'] < self.cache_duration:
                        print("Loaded from cache.")
                        return cache['data']
        except json.JSONDecodeError:
            print("Error decoding JSON from cache. Fetching from API...")
        return None

    def _save_cache(self, data):
        """Save data to cache with timestamp."""
        cache = {
            'timestamp': time.time(),
            'data': data
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f)

    def fetch_players(self, use_cache=True, invalidate_cache=False):
        """
        Fetch NBA players, handling caching and potential errors.

        Args:
            use_cache (bool): Whether to use cached data if available.
            invalidate_cache (bool): Whether to force cache invalidation.

        Returns:
            pandas.DataFrame: DataFrame containing players' information.
        """
        if use_cache and not invalidate_cache:
            cached_data = self._load_cache()
            if cached_data:
                return pd.DataFrame(cached_data)

        try:
            params = {
                'LeagueID': '00',
                'Season': self._get_current_season(),
                'IsOnlyCurrentSeason': '1'
            }
            response = self.session.get(self.base_url, params=params, headers=self.headers, timeout=15)
            response.raise_for_status()

            data = response.json()

            headers = data['resultSets'][0]['headers']
            players_data = data['resultSets'][0]['rowSet']

            players = []
            for player in players_data:
                player_dict = dict(zip(headers, player))
                if player_dict['TO_YEAR'] == self._get_current_season()[:4]:
                    players.append({
                        'player_id': player_dict['PERSON_ID'],
                        'full_name': player_dict['DISPLAY_FIRST_LAST'],
                        'team_id': player_dict['TEAM_ID'],
                        'team': player_dict['TEAM_NAME'] if player_dict['TEAM_ID']!=0 else "Free Agent",
                        'is_active': True
                    })

            df = pd.DataFrame(players)
            self._save_cache(df.to_dict('records'))  # Save DataFrame as list of dicts
            return df

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
        except KeyError as e:
            print(f"API response format error (missing key): {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        if use_cache:
            print("Using cached data as fallback (if available).")
            cached_data = self._load_cache()
            if cached_data:
                return pd.DataFrame(cached_data)

        return pd.DataFrame()

    def _get_current_season(self):
        """Determine the current NBA season (e.g., "2023-24")."""
        now = datetime.now()
        year = now.year
        if now.month >= 10:
            return f"{year}-{str(year + 1)[2:]}"
        else:
            return f"{year - 1}-{str(year)[2:]}"

# --- Load environment variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Configure Gemini API ---
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# --- Instantiate NBA Players Fetcher ---
nba_fetcher = NBAPlayersFetcher()

# --- Initialize variables in session_state ---
if 'chat' not in st.session_state:
    st.session_state.chat = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'youtube_url' not in st.session_state:
    st.session_state.youtube_url = None
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""
if 'selected_prompt' not in st.session_state:
    st.session_state.selected_prompt = None
if 'corrected_transcript' not in st.session_state:
    st.session_state.corrected_transcript = None
if 'player_master_list' not in st.session_state:
    st.session_state.player_master_list = []

# --- Fetch and update player_master_list (on app start) ---
if not st.session_state.player_master_list:
    with st.spinner("Updating player list..."):
        try:
            df = nba_fetcher.fetch_players()
            st.session_state.player_master_list = df['full_name'].tolist()
            st.session_state.player_master_list.sort()
            st.success("Player list updated!")
        except Exception as e:
            st.error(f"Failed to update player list: {e}")

# --- Fuzzy Matching Function ---
def find_closest_player_name(transcript_name, player_master_list):
    best_match, score = process.extractOne(transcript_name, player_master_list, scorer=fuzz.token_set_ratio)
    if score >= 80:
        return best_match
    else:
        return None

# --- Transcript Handling Functions ---
def get_transcript(youtube_url):
    try:
        video_id = extract_video_id(youtube_url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        formatted_transcript = formatter.format_transcript(transcript)
        cleaned_transcript = clean_transcript(formatted_transcript)
        st.session_state.corrected_transcript = correct_player_names(cleaned_transcript, st.session_state.player_master_list)
        return st.session_state.corrected_transcript
    except YouTubeTranscriptApi.NoTranscriptFound:
        return "No transcript found for this video."
    except Exception as e:
        return f"An error occurred: {e}"

def extract_video_id(youtube_url):
    try:
        yt = YouTube(youtube_url)
        return yt.video_id
    except Exception as e:
        st.error(f"Error extracting video ID: {e}")
        return None

def clean_transcript(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def correct_player_names(transcript, player_master_list):
    words = transcript.split()
    corrected_words = []
    for word in words:
        if word.istitle():
            possible_player_name = find_closest_player_name(word, player_master_list)
            if possible_player_name:
                corrected_words.append(possible_player_name)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

# --- Gemini Chat Initialization ---
def initialize_gemini_chat(transcript):
    chat = model.start_chat(history=[
        {
            "role": "user",
            "parts": [f"This is the transcript of a YouTube video:\n\n{transcript}"]
        },
        {
            "role": "model",
            "parts": ["Okay, I understand. I have the context from the transcript. Ask me anything about it."]
        }
    ])
    return chat

# --- Fantasy Basketball Prompts ---
prompt_templates = [
    "Which players mentioned in the video are good waiver wire pickups?",
    "Based on this video, who are some buy-low or sell-high candidates?",
    "What are the injury updates discussed in the video, and how do they impact player value?",
    "Does this video suggest any players I should drop from my roster?",
    "Summarize the overall strategy discussed in the video (e.g., streaming, punting categories).",
    "Generate a list of players mentioned in this video, categorized by position and projected value."
]

def prompt_button_clicked(prompt):
    st.session_state.selected_prompt = prompt
    st.session_state.user_question = ""

# --- Streamlit UI ---
st.title("YouTube Transcript Analyzer with Gemini")

# Manual Player List Update Button
if st.button("Update Player List"):
    with st.spinner("Updating player list..."):
        try:
            df = nba_fetcher.fetch_players(invalidate_cache=True)
            st.session_state.player_master_list = df['full_name'].tolist()
            st.session_state.player_master_list.sort()
            st.success("Player list updated!")
        except Exception as e:
            st.error(f"Failed to update player list: {e}")

st.session_state.youtube_url = st.text_input("Enter YouTube Video URL:", value=st.session_state.youtube_url if st.session_state.youtube_url else "")

if st.button("Get Transcript"):
    if st.session_state.youtube_url:
        with st.spinner("Fetching transcript..."):
            st.session_state.transcript = get_transcript(st.session_state.youtube_url)
            st.subheader("Transcript:")
            st.text_area("Transcript", st.session_state.corrected_transcript, height=300)
    else:
        st.warning("Please enter a YouTube video URL.")

if st.session_state.transcript:
    if st.button("Load Transcript into Gemini"):
        with st.spinner("Loading transcript into Gemini..."):
            st.session_state.chat = initialize_gemini_chat(st.session_state.corrected_transcript)
            st.success("Transcript loaded into Gemini. You can now ask questions!")

if st.session_state.chat is not None:
    st.subheader("Ask Gemini about the video:")

    for prompt in prompt_templates:
        st.button(prompt, on_click=prompt_button_clicked, args=(prompt,))

    st.session_state.user_question = st.text_input("Your Question:", value=st.session_state.user_question)

    if st.button("Ask"):
        if st.session_state.selected_prompt:
            user_question = st.session_state.selected_prompt
            st.session_state.selected_prompt = None
        else:
            user_question = st.session_state.user_question

        if user_question:
            with st.spinner("Gemini is thinking..."):
                st.session_state.chat.send_message(user_question)
                st.write(f"Gemini: {st.session_state.chat.last.text}")
        else:
            st.warning("Please enter a question or select a prompt.")