from datetime import datetime
from pathlib import Path

import streamlit as st
from config import get_page_config, get_whisper_settings, save_whisper_settings
from core import MediaManager





st.set_page_config(**get_page_config())


# Session states
# --------------
# Set session state to toggle list & detail view
if "list_mode" not in st.session_state:
    st.session_state.list_mode = True
    st.session_state.selected_media = None
    st.session_state.selected_media_offset = 0

# Add whisper settings to session state
if "whisper_params" not in st.session_state:
    st.session_state.whisper_params = get_whisper_settings()

if "media_manager" not in st.session_state:
    st.session_state.media_manager = MediaManager()

# Alias for session state media manager
media_manager = st.session_state.media_manager







