import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time
import numpy as np
import cv2

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.start_time = time.time()
        self.frames = []
        self.last_capture_time = self.start_time

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        # 1秒ごとにフレームをキャプチャ
        if current_time - self.last_capture_time >= 1:
            if len(self.frames) < 30:
                self.frames.append(img)
                self.last_capture_time = current_time

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("1秒ごとに30秒間分の画像表示")

    # WebRTC 設定を定義
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    # VideoProcessorを使用してwebrtc_streamerを設定
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIGURATION)

    # 撮影開始ボタン
    if st.sidebar.button('撮影開始'):
        if ctx.video_processor:
            processor = ctx.video_processor
            st.write("画像をキャプチャ中...")
            # 30秒待つ
            time.sleep(30)
            # キャプチャしたフレームを表示
            for i, frame in enumerate(processor.frames):
                st.image(frame, channels="BGR", caption=f"Frame {i+1}")

if __name__ == "__main__":
    main()
