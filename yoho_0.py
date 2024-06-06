import cv2 as cv
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
import copy
import streamlit as st
import time
import math
import matplotlib.pyplot as plt
import requests

# 画像から姿勢推定　＆　推論部分の関数を定義
def run_inference(onnx_session, input_size, image):
    if image is None:
        return None, None # 画像がNoneの場合は、関数から戻る
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
input_size=192
keypoint_score_th=0.3

# 推論結果を可視化する関数
def draw_debug_with_line_lengths(image, keypoint_score_th, keypoints, scores):
    debug_image = copy.deepcopy(image)
    
    # キーポイント間の接続（左耳から左肩・右耳から右肩）定義。各接続は、2つのキーポイントのインデックスと、その接続線の色を定義。
    connect_list = [
        [3, 5, (255, 0, 0)], # left ear → left shoulder
        [4, 6, (0, 0, 255)], # right ear → right shoulder
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

            distance = np.sqrt((point01[0] - point02[0])**2 + (point01[1] - point02[1])**2)
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
        left_distance = np.sqrt((left_ear[0] - left_shoulder[0])**2 + (left_ear[1] - left_shoulder[1])**2)
    if (scores[right_ear_index] > keypoint_score_th and
        scores[right_shoulder_index] > keypoint_score_th):
        right_ear = keypoints[right_ear_index]
        right_shoulder = keypoints[right_shoulder_index]
        right_distance = np.sqrt((right_ear[0] - right_shoulder[0])**2 + (right_ear[1] - right_shoulder[1])**2)

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
        # left_eye = keypoints[0][left_eye_index]
        # right_eye = keypoints[0][right_eye_index]
        # nose = keypoints[0][nose_index]
        left_eye = keypoints[left_eye_index]
        right_eye = keypoints[right_eye_index]
        nose = keypoints[nose_index]
        
        # 三角形の3辺の長さを計算
        side1 = math.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)
        side2 = math.sqrt((left_eye[0] - nose[0])**2 + (left_eye[1] - nose[1])**2)
        side3 = math.sqrt((right_eye[0] - nose[0])**2 + (right_eye[1] - nose[1])**2)

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
    
# Streamlitアプリケーションのメイン部分
def main():
    st.title("肩こり予報アプリ")

    st.sidebar.markdown("""★撮影ボタンを押す前のポイント★""")
    st.sidebar.markdown("""【1】肩の力を抜いてリラックスし、耳と肩の距離を遠ざけましょう""")
    st.sidebar.markdown("""【2】目の奥で軽く後頭部を押し、あごを軽く引きましょう""")

     # カメラのアクセスを要求するHTMLとJavaScript
    camera_access_code = """
    <script>
    function requestCameraAccess() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({video: true}).then(function(stream) {
                var video = document.createElement('video');
                video.srcObject = stream;
                video.width = 640;
                video.height = 480;
                document.body.appendChild(video);
                video.play();
            }).catch(function(err) {
                console.log("An error occurred: " + err);
            });
        } else {
            alert("Your browser does not support media devices.");
        }
    }
    window.onload = requestCameraAccess;
    </script>
    """

    # HTMLとJavaScriptをStreamlitに表示
    st.markdown(camera_access_code, unsafe_allow_html=True)
    
    # 撮影ボタンを作成
    if st.sidebar.button('撮影開始'):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            st.error("カメラを開けませんでした。カメラが接続されているか確認してください。")
            return
            
        start_time = time.time()
        frame_list = []
        keypoints_list = []
        scores_list = []
        frame_distances = []
        frame_triangle_areas = []
        initial_distance = None
        initial_triangle_area = None

        while time.time() - start_time < 30:
            ret, frame = cap.read()
            if not ret:
                break

            if int(time.time() - start_time) % 1 == 0:
                keypoints, scores = run_inference(onnx_session, input_size, frame)
                keypoints_list.append(keypoints)
                scores_list.append(scores)
                debug_frame = draw_debug_with_line_lengths(frame, keypoint_score_th, keypoints, scores)
                frame_list.append(debug_frame)

                distance = calculate_ear_shoulder_distance(keypoints, scores)
                frame_distances.append(distance)
                if initial_distance is None:
                    initial_distance = distance

                triangle_area = calculate_triangle_area(keypoints, scores)
                frame_triangle_areas.append(triangle_area)
                if initial_triangle_area is None and triangle_area is not None:
                    initial_triangle_area = triangle_area
                
            time.sleep(1) 

        cap.release()
        

        # 耳と肩の距離の変化率を計算
        final_distances = [calculate_ear_shoulder_distance(keypoints, scores) for keypoints, scores in zip(keypoints_list, scores_list)]
        final_distance = np.mean(final_distances)
        if initial_distance is not None and final_distance is not None:
            change_ratio = (final_distance - initial_distance) / initial_distance * 100
        else:
            change_ratio = None
               
        # 撮影開始時の三角形の面積を計算
        initial_triangle_area = None
        if keypoints_list and scores_list:
           initial_triangle_area = calculate_triangle_area(keypoints_list[0], scores_list[0])
        
        # 三角形の面積の変化率を計算
        final_triangle_areas = [calculate_triangle_area(keypoints, scores) for keypoints, scores in zip(keypoints_list, scores_list)]
        final_triangle_areas = [area for area in final_triangle_areas if area is not None]


        if final_triangle_areas:
            final_triangle_area = np.mean(final_triangle_areas)
            if initial_triangle_area is not None:
                triangle_area_change_ratio = (final_triangle_area - initial_triangle_area) / initial_triangle_area * 100
            else:
                triangle_area_change_ratio = None
        else:
            final_triangle_area = None
            triangle_area_change_ratio = None

     # 肩こり予報を表示
        if change_ratio is not None and triangle_area_change_ratio is not None:
         if change_ratio <= -20 or triangle_area_change_ratio >= 10:
          st.markdown("""
            <div style="font-size:17px;">
                【肩こり予報：高】あなたの姿勢は肩こりになるリスクが高いです
            </div>
            <div style="font-size:14px;">
                 <br>&#42;～&#42;ちょっと休憩して、リラックスしましょう&#42;～&#42;<br>
                 <br>   <br>
                 あなたの姿勢の推移です↓
                 <br>   <br>
            </div>
        """, unsafe_allow_html=True)

         elif change_ratio <= -10 and change_ratio > -20 or triangle_area_change_ratio >= 5 and triangle_area_change_ratio < 10:
          st.markdown("""
            <div style="font-size:17px;">
                【肩こり予報：中】   あなたの姿勢は肩こりになるリスクが中程度です
            </div>
            <div style="font-size:14px;">
                 <br>&#42;～&#42;ちょっと休憩して、リラックスしましょう&#42;～&#42;<br>
                 <br>   <br>
                 あなたの姿勢の推移です↓
                 <br>   <br>
            </div>
        """, unsafe_allow_html=True)
          
        else:
         st.markdown("""
           <div style="font-size:17px;">
            【肩こり予報：低】   あなたの姿勢は肩こりになるリスクが低いです
           </div>
           <div style="font-size:14px;">
                <br>&#42;～&#42;～&#42; このままリラックスした状態で すごしてください &#42;～&#42;～&#42; <br>
                     <br>   <br>
                 あなたの姿勢の推移です↓
                 <br>   <br>
            </div>
        """, unsafe_allow_html=True)        
        
        # グラフを描画
        fig = plot_graphs(frame_distances, frame_triangle_areas, initial_distance, initial_triangle_area)
        st.pyplot(fig)

        if not frame_distances:
           st.warning("耳と肩の距離のデータがありません。")
        if not frame_triangle_areas:
           st.warning("三角形の面積のデータがありません。")
        
        # 画像をサイドバーに並べて表示
        num_cols = 10  # 1行に表示する画像の数
        num_rows = (len(frame_list) + num_cols - 1) // num_cols  # 必要な行数
        for row in range(num_rows):
            cols = st.sidebar.columns(num_cols)  # サイドバー内で列を作成
            for col_idx, col in enumerate(cols):
                idx = row * num_cols + col_idx
                if idx < len(frame_list):
                   with col:
                      col.image(frame_list[idx], channels="BGR")
                      col.markdown(f'<span style="font-size:10px;"> {idx}</span>', unsafe_allow_html=True)                                                     
                                              
if __name__ == "__main__":
    main()   
    
    
    

