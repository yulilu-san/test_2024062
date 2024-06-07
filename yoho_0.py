import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import threading
from datetime import datetime
import time

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.timestamps = []
        self.capture_flag = False
        self.capture_interval = 1  # capture interval in seconds
        self.last_capture_time = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.capture_flag:
            current_time = time.time()
            if self.last_capture_time is None or current_time - self.last_capture_time >= self.capture_interval:
                self.frames.append(img)
                self.timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                self.last_capture_time = current_time
                if len(self.frames) >= 3:
                    self.capture_flag = False  # Stop capturing after 3 frames
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def capture_frames(self):
        self.frames = []
        self.timestamps = []
        self.capture_flag = True
        self.last_capture_time = None
        while self.capture_flag:
            time.sleep(0.1)  # short sleep to yield control

    def get_frames(self):
        return self.frames, self.timestamps

def main():
    st.title("WebRTC Photo Capture")

    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        if st.button("Start Capture"):
            capture_thread = threading.Thread(target=ctx.video_processor.capture_frames)
            capture_thread.start()

            # スレッドが終了するのを待つ
            capture_thread.join()

            # キャプチャしたフレームを取得して表示
            frames, timestamps = ctx.video_processor.get_frames()
            if frames:
                st.write("Captured Images:")
                for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                    st.image(frame, caption=f"Captured Image {idx + 1} at {timestamp}")

if __name__ == "__main__":
    main()





# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
# import av
# import threading
# import time

# class VideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.frames = []
#         self.capture_flag = False

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         if self.capture_flag:
#             print("★3")
#             self.frames.append(img)
#             if len(self.frames) >= 3:
#                 self.capture_flag = False  # Stop capturing after 3 frames
#         return av.VideoFrame.from_ndarray(img, format="bgr24")

#     def capture_frames(self):
#         self.frames = []
#         self.capture_flag = True
#         for _ in range(3):
#             if not self.capture_flag:
#                 break
#             time.sleep(1)
#         self.capture_flag = False

#     def get_frames(self):
#         return self.frames

# def main():
#     st.title("WebRTC Photo Capture")

#     rtc_config = RTCConfiguration(
#         {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
#     )

#     ctx = webrtc_streamer(
#         key="example",
#         mode=WebRtcMode.SENDRECV,
#         rtc_configuration=rtc_config,
#         video_processor_factory=VideoProcessor,
#         media_stream_constraints={"video": True, "audio": False},
#     )

#     if ctx.video_processor:
#         if st.button("Start Capture"):
#             capture_thread = threading.Thread(target=ctx.video_processor.capture_frames)
#             capture_thread.start()

#             # スレッドが終了するのを待つ
#             capture_thread.join()

#             # キャプチャしたフレームを取得して表示
#             frames = ctx.video_processor.get_frames()
#             print("★1")
#             if frames:
#                 print("★2")
#                 st.write("Captured Images:")
#                 for idx, frame in enumerate(frames):
#                     st.image(frame, caption=f"Captured Image {idx + 1}")

# if __name__ == "__main__":
#     main()







# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
# import av

# class VideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.frames = []

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         self.frames.append(img)
#         return av.VideoFrame.from_ndarray(img, format="bgr24")

#     def get_frames(self):
#         return self.frames

# def main():
#     st.title("WebRTC Video Capture")

#     rtc_config = RTCConfiguration(
#         {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
#     )

#     ctx = webrtc_streamer(
#         key="example",
#         mode=WebRtcMode.SENDRECV,
#         rtc_configuration=rtc_config,
#         video_processor_factory=VideoProcessor,
#         media_stream_constraints={"video": True, "audio": False},
#     )

#     if ctx.video_processor:
#         if st.button("Save Video"):
#             frames = ctx.video_processor.get_frames()
#             if frames:
#                 # 保存するコード（例えば、ファイルとして保存するなど）
#                 st.write(f"Captured {len(frames)} frames")
#             else:
#                 st.warning("No frames captured")

# if __name__ == "__main__":
#     main()



# import logging
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
# import av
# import time

# logging.basicConfig(level=logging.WARN)

# class VideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         self.frames = []
#         self.last_capture_time = time.time()

#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         img = frame.to_ndarray(format="bgr24")
#         current_time = time.time()

#         if current_time - self.last_capture_time >= 1 and len(self.frames) < 10:
#             self.frames.append(img)
#             self.last_capture_time = current_time
#             logging.debug(f"Captured frame {len(self.frames)} at {current_time}")

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# def main():
#     st.title("1秒ごとに10秒間分の画像表示")

#     # WebRTC 設定を定義
#     RTC_CONFIGURATION = RTCConfiguration({
#         "iceServers": [
#             {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302", "stun:stun2.l.google.com:19302"]}
#         ]
#     })

#     # VideoProcessorを使用してwebrtc_streamerを設定
#     ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIGURATION)

#     # 撮影開始ボタン
#     if st.sidebar.button('撮影開始'):
#         if ctx.video_processor:
#             processor = ctx.video_processor
#             st.write("画像をキャプチャ中...")
#             # 10秒待つ
#             time.sleep(10)
#             # キャプチャしたフレームを表示
#             for i, frame in enumerate(processor.frames):
#                 st.image(frame, channels="BGR", caption=f"Frame {i+1}")

# if __name__ == "__main__":
#     main()
