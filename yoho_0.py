import cv2 as cv
import numpy as np
import onnxruntime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import matplotlib.pyplot as plt  # 追加

# 画像から姿勢推定＆推論部分の関数を定義
def run_inference(onnx_session, input_size, image):
    if image is None:
        return None, None  # 画像がNoneの場合は、関数から戻る
    image_width, image_height = image.shape[1], image.shape[0]
    # 画像の前処理
    input_image = cv.resize(image, dsize=(input_size, input_size))
    input_image = cv.cvtColor(input_image, cv.COLOR_RGB2BGR)
    input_image = input_image.reshape(-1, input_size, input_size, 3)
    input_image = input_image.astype('float32')
    # ONNXモデルを使用し画像データに対する推論・その結果を処理
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: input_image})
    keypoints_with_scores = outputs[0]
    keypoints_with_scores = np.squeeze(keypoints_with_scores)
    # キーポイントの座標とそれらのスコアを計算
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append((keypoint_x, keypoint_y))
        scores.append(score)

    return keypoints, scores

# モデルの読み込みとパラメータの設定
onnx_session = onnxruntime.InferenceSession('./edge-pose-estimation/model/model_float32.onnx')
input_size = 192
keypoint_score_th = 0.3

# 推論結果を可視化する関数
def draw_debug_with_line_lengths(image, keypoint_score_th, keypoints, scores):
    debug_image = copy.deepcopy(image)

    # キーポイント間の接続（左耳から左肩・右耳から右肩）定義。各接続は、2つのキーポイントのインデックスと、その接続線の色を定義。
    connect_list = [
        [3, 5, (255, 0, 0)],  # left ear → left shoulder
        [4, 6, (0, 0, 255)],  # right ear → right shoulder
    ]

    # 左目、右目、鼻のキーポイントのインデックス
    left_eye_index = 1
    right_eye_index = 2
    nose_index = 0

    # 左目、右目、鼻のキーポイントが有効な場合、三角形を描画する
    if (scores[left_eye_index] > keypoint_score_th and
            scores[right_eye_index] > keypoint_score_th and
            scores[nose_index] > keypoint_score_th):
        left_eye = keypoints[left_eye_index]
        right_eye = keypoints[right_eye_index]
        nose = keypoints[nose_index]

        # 三角形の線を描画
        cv.line(debug_image, left_eye, right_eye, (0, 255, 0), 2)
        cv.line(debug_image, left_eye, nose, (0, 255, 0), 2)
        cv.line(debug_image, right_eye, nose, (0, 255, 0), 2)

    for (index01, index02, color) in connect_list:
        if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
            point01 = keypoints[index01]
            point02 = keypoints[index02]
            cv.line(debug_image, point01, point02, color, 2)

            distance = np.sqrt((point01[0] - point02[0]) ** 2 + (point01[1] - point02[1]) ** 2)
            midpoint = ((point01[0] + point02[0]) // 2, (point01[1] + point02[1]) // 2)
            cv.putText(debug_image, f"{distance:.2f}", midpoint, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for keypoint, score in zip(keypoints, scores):
        if score > keypoint_score_th:
            cv.circle(debug_image, keypoint, 3, (0, 255, 0), -1)

    return debug_image

# 耳と肩の距離を計算する関数
def calculate_ear_shoulder_distance(keypoints, scores):
    left_ear_index = 3
    right_ear_index = 4
    left_shoulder_index = 5
    right_shoulder_index = 6

    left_distance = 0
    right_distance = 0

    if (scores[left_ear_index] > keypoint_score_th and
            scores[left_shoulder_index] > keypoint_score_th):
        left_ear = keypoints[left_ear_index]
        left_shoulder = keypoints[left_shoulder_index]
        left_distance = np.sqrt((left_ear[0] - left_shoulder[0]) ** 2 + (left_ear[1] - left_shoulder[1]) ** 2)
    if (scores[right_ear_index] > keypoint_score_th and
            scores[right_shoulder_index] > keypoint_score_th):
        right_ear = keypoints[right_ear_index]
        right_shoulder = keypoints[right_shoulder_index]
        right_distance = np.sqrt((right_ear[0] - right_shoulder[0]) ** 2 + (right_ear[1] - right_shoulder[1]) ** 2)

    if left_distance is None and right_distance is None:
        return 0
    elif left_distance is None:
        return right_distance
    elif right_distance is None:
        return left_distance
    else:
        return (left_distance + right_distance) / 2

# 両目と鼻を結んだ三角形面積を計算する関数
def calculate_triangle_area(keypoints, scores):
    left_eye_index = 1
    right_eye_index = 2
    nose_index = 0

    if (scores[left_eye_index] > keypoint_score_th and
            scores[right_eye_index] > keypoint_score_th and
            scores[nose_index] > keypoint_score_th):
        left_eye = keypoints[left_eye_index]
        right_eye = keypoints[right_eye_index]
        nose = keypoints[nose_index]

        # 三角形の3辺の長さを計算
        side1 = math.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)
        side2 = math.sqrt((left_eye[0] - nose[0]) ** 2 + (left_eye[1] - nose[1]) ** 2)
        side3 = math.sqrt((right_eye[0] - nose[0]) ** 2 + (right_eye[1] - nose[1]) ** 2)

        # ヘロンの公式で三角形の面積を求める
        semi_perimeter = (side1 + side2 + side3) / 2
        area = math.sqrt(semi_perimeter * (semi_perimeter - side1) * (semi_perimeter - side2) * (semi_perimeter - side3))

        return area
    else:
        return None

# グラフを描画する関数
def plot_graphs(frame_distances, frame_triangle_areas, initial_distance, initial_triangle_area):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # 耳と肩の距離のグラフ
    if frame_distances:
        axs[0].plot(frame_distances)
    if initial_distance is not None:
        axs[0].axhline(y=initial_distance, color='r', linestyle='--', label='Initial Distance')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Ear to Shoulder Distance')
    axs[0].legend()

    # 三角形の面積のグラフ
    if frame_triangle_areas:
        axs[1].plot(frame_triangle_areas)
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Triangle Area')
    if initial_triangle_area is not None:
        axs[1].axhline(y=initial_triangle_area, color='r', linestyle='--', label='Initial Area')
    axs[1].legend()

    return fig

# VideoProcessorBaseを継承してカメラ映像を処理するクラスを定義
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.keypoint_score_th = keypoint_score_th
        self.input_size = input_size
        self.onnx_session = onnx_session

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        keypoints, scores = run_inference(self.onnx_session, self.input_size, img)
        img = draw_debug_with_line_lengths(img, self.keypoint_score_th, keypoints, scores)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlitアプリケーションのメイン部分
def main():
    st.title("肩こり予報アプリ")

    st.sidebar.markdown("""★撮影ボタンを押す前のポイント★""")
    st.sidebar.markdown("""【1】肩の力を抜いてリラックスし、耳と肩の距離を遠ざけましょう""")
    st.sidebar.markdown("""【2】目の奥で軽く後頭部を押し、あごを軽く引きましょう""")

    # webrtc_streamerを使ってカメラ映像をリアルタイム処理
    webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

    # グラフ描画の準備
    frame_distances = []
    frame_triangle_areas = []
    initial_distance = None
    initial_triangle_area = None

    # グラフを描画
    fig = plot_graphs(frame_distances, frame_triangle_areas, initial_distance, initial_triangle_area)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
