import streamlit as st
# カメラの画像をキャプチャ
picture = st.camera_input("Take a picture")
if picture:
    st.image(picture)
