import streamlit as st
import cv2
import torch
from utils.hubconf import custom
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from model_utils import get_yolo, color_picker_fn, get_system_stat
from ultralytics import YOLO

# Add view
# ---------
with st.sidebar.expander("➕ &nbsp; Add Media", expanded=False):
    # # Render media type selection on the sidebar & the form
    source_type = st.radio("Media Source", ["YouTube", "Upload"], label_visibility="collapsed")
    with st.form("input_form"):
        if source_type == "YouTube":
            youtube_url = st.text_input("Youtube video or playlist URL")
        elif source_type == "Upload":
            input_files = st.file_uploader(
                "Add one or more files", type=["mp4", "avi", "mov", "mkv", "mp3", "wav"], accept_multiple_files=True
            )