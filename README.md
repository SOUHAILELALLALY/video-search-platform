# video-search-platform
Video Search allows users to find relevant video clips using text or image queries. The project leverages **CLIP embeddings** and **FAISS** for efficient similarity search and provides a **Streamlit interface** for interactive exploration.

## Features

    - Video to Clip Splitting: Automatically splits videos into short clips.
    
    - Frame Extraction: Extracts the middle frame from each clip for embedding.
    
    - Text & Image Embeddings: Uses OpenAI’s CLIP model to embed both images and text.
    
    - FAISS Indexing: Efficient similarity search using vector embeddings.
    
    - Interactive Search: Streamlit web interface for querying by text or image.
    
    - Supports Multiple Video Formats: .mp4, .mov, .mkv, .avi.

## Requirements

    - Python 3.9+
    
    - FFmpeg installed and available in PATH
    
    - PyTorch
    
    - Transformers
    
    - PIL (Pillow)
    
    - NumPy
    
    - FAISS
    
    - Streamlit
    
    - tqdm

## Usage

**1. Build the Index**

This step processes all videos in **sample_videos/**, splits them into clips, extracts frames, generates embeddings, and builds a FAISS index.
python preprocess.py

**2. Run the Streamlit App**

After building the index, start the web interface:
streamlit run app_streamlit.py

    - Enter a text query describing the scene you want to find.
    
    - Or upload an image to find similar clips.
    
    - Results will display the top-ranked clips


