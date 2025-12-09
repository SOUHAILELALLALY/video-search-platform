import streamlit as st
import faiss
import pickle
import numpy as np
from PIL import Image
from io import BytesIO

from utils import embed_text, embed_image

INDEX_PATH = "index.faiss"
METADATA_PATH = "metadata.pkl"
TOP_K = 3

@st.cache_resource
def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search_by_embedding(index, metadata, query_emb, top_k=TOP_K):
    # ensure numpy float32 and shape (1,d)
    q = np.array(query_emb, dtype=np.float32).reshape(1, -1)
    D, I = index.search(q, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def main():
    st.title("Video Search — Text or Image to Clip")

    index, metadata = load_index()

    col1, col2 = st.columns([3,1])
    with col1:
        text_query = st.text_input("Describe the scene (text query)")
        img_file = st.file_uploader("Or upload an image to search (optional)", type=["jpg","jpeg","png"])
        if st.button("Search"):
            if img_file is not None:
                # Image search takes precedence
                image = Image.open(img_file).convert("RGB")
                emb = embed_image(image)
                results = search_by_embedding(index, metadata, emb)
                st.subheader("Results (image query)")
            elif text_query:
                emb = embed_text(text_query)
                results = search_by_embedding(index, metadata, emb)
                st.subheader("Results (text query)")
            else:
                st.warning("Provide a text query or upload an image.")
                return

            if not results:
                st.info("No matches found.")
                return

            for i, r in enumerate(results):
                st.write(f"**Rank {i+1} — score={r['score']:.4f}**")
                st.write(r['clip_path'])
                try:
                    with open(r['clip_path'], 'rb') as f:
                        video_bytes = f.read()
                    st.video(video_bytes)
                except Exception as e:
                    st.error(f"Failed to open video: {e}")

    with col2:
        st.markdown("### Tips")
        st.markdown("- Short, descriptive text queries work best.\n- Upload an image of the scene you want to find.\n- If results look wrong, consider rebuilding the index with more frames per clip.")

if __name__ == '__main__':
    main()
