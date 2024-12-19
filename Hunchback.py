import cv2
import math as m
import mediapipe as mp
import streamlit as st
import sqlite3
import pandas as pd
import time
import pygame
import datetime
import matplotlib.pyplot as plt

# 初始化 Mediapipe 姿勢模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, static_image_mode=False, min_tracking_confidence=0.5)

# 設定 Streamlit 頁面標題
st.title("姿勢偵測與分析系統")
FRAME_WINDOW = st.image([])

# 初始化 SQLite 資料庫
def init_db():
    conn = sqlite3.connect("posture.db")
    cursor = conn.cursor()

    # 創建新的資料表，torso_inclination欄位為REAL（小數）
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS posture_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cva_angle REAL,
            torso_inclination REAL,
            posture_status TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 初始化 SQLite 資料庫，新增音效觸發事件表格
def init_audio_event_db():
    conn = sqlite3.connect("posture.db")
    cursor = conn.cursor()

    # 創建音效觸發事件資料表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_event_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 儲存音效觸發事件
def log_audio_event():
    conn = sqlite3.connect("posture.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO audio_event_data (timestamp)
        VALUES (datetime('now', 'localtime'))
    ''')
    conn.commit()
    conn.close()


# 儲存姿勢數據到資料庫 (不再包含 alert_issued_shifted)
def log_posture(cva_angle, torso_inclination, posture_status):
    conn = sqlite3.connect("posture.db")
    cursor = conn.cursor()
    cursor.execute(''' 
        INSERT INTO posture_data (timestamp, cva_angle, torso_inclination, posture_status)
        VALUES (datetime('now', 'localtime'), ?, ?, ?)
    ''', (cva_angle, torso_inclination, posture_status))
    conn.commit()
    conn.close()

# 計算CVA角度
def calculate_cva_angle(ear_x, ear_y, shoulder_x, shoulder_y):
    try:
        delta_x = shoulder_x - ear_x
        delta_y = ear_y - shoulder_y
        angle = m.degrees(m.atan2(delta_y, delta_x))
        return abs(angle)
    except ZeroDivisionError:
        return 90  # 預設安全值

# 計算兩點形成的夾角，並返回小數
def findAngle(x1, y1, x2, y2):
    try:
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * abs(y1)))
        return theta * 180 / m.pi  # 返回小數
    except (ValueError, ZeroDivisionError):
        return 90  # 預設安全值

# 初始化資料庫
init_db()
init_audio_event_db()

# 初始化音效相關設定
warning_sound = "Announcement_sound_effect.mp3"
pygame.mixer.init()
is_playing_warning_sound = False

# 播放警告音效並紀錄事件
def play_warning_sound():
    global is_playing_warning_sound
    if not is_playing_warning_sound:
        pygame.mixer.music.load(warning_sound)
        pygame.mixer.music.play(loops=-1)
        is_playing_warning_sound = True
        log_audio_event()  # 記錄音效觸發事件


# 停止警告音效
def stop_warning_sound():
    global is_playing_warning_sound
    if is_playing_warning_sound:
        pygame.mixer.music.stop()
        is_playing_warning_sound = False

# Main logic
detect_posture = st.sidebar.button("偵測姿勢")
cap = None

if detect_posture:
    if cap and cap.isOpened():
        cap.release()
    FRAME_WINDOW.empty()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    bad_posture_start_time = None
    posture_alert_time = 5  # Alert time for bad posture
    showed_warning = False
    warning_message = None
    previous_posture_status = "Good"  # Initialize as Good
    last_log_time = None  # Initialize for limiting log frequency

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.write("Cannot capture video frame")
            break

        h, w = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(frame)


        # 根據姿勢偏差顯示箭頭調整方向
        def draw_adjustment_arrow(frame, cva_angle, torso_inclination, ear_x, ear_y, shoulder_x, shoulder_y, hip_x,
                                  hip_y):
            arrow_color = (0, 0, 255)  # 紅色箭頭表示錯誤的姿勢
            arrow_thickness = 2

            # 根據CVA角度顯示箭頭
            if cva_angle > 10:  # 如果CVA角度過大，顯示調整箭頭
                if ear_y > shoulder_y:  # 頭部過低，箭頭指向上方
                    cv2.arrowedLine(frame, (ear_x, ear_y), (ear_x, ear_y - 50), arrow_color, arrow_thickness)
                else:  # 頭部過高，箭頭指向下方
                    cv2.arrowedLine(frame, (ear_x, ear_y), (ear_x, ear_y + 50), arrow_color, arrow_thickness)

            # 根據躯干倾斜角度顯示箭頭
            if torso_inclination > 10:  # 如果躯干倾斜角度過大，顯示調整箭頭
                cv2.arrowedLine(frame, (hip_x, hip_y), (hip_x, hip_y - 50), arrow_color, arrow_thickness)

        if keypoints.pose_landmarks:
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark

            # Get coordinates of ear and shoulder
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # Draw the detected points
            cv2.circle(frame, (l_ear_x, l_ear_y), 5, (0, 255, 0), -1)  # Green circle for ear
            cv2.circle(frame, (l_shldr_x, l_shldr_y), 5, (255, 0, 0), -1)  # Blue circle for shoulder
            cv2.circle(frame, (l_hip_x, l_hip_y), 7, (127, 255, 0), -1)
            cv2.line(frame, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (127, 255, 0), 2)
            cv2.line(frame, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (127, 255, 0), 2)

            # Calculate CVA angle
            cva_angle = calculate_cva_angle(l_ear_x, l_ear_y, l_shldr_x, l_shldr_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            # Posture detection logic
            if cva_angle >= 50 and torso_inclination < 15:
                posture_status = "Good"
                cv2.putText(frame, "Good Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (127, 255, 0), 2)
                stop_warning_sound()
                bad_posture_start_time = None
                showed_warning = False
                if warning_message:
                    warning_message.empty()
                    warning_message = None
            else:
                posture_status = "Bad"
                if bad_posture_start_time is None:
                    bad_posture_start_time = time.time()
                if time.time() - bad_posture_start_time >= posture_alert_time and not showed_warning:
                    if warning_message is None:
                        warning_message = st.warning("你已經保持不良姿勢超過5秒鐘了！")
                    play_warning_sound()
                    showed_warning = True
                else:
                    showed_warning = False

                cv2.putText(frame, "Bad Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)

            # 顯示箭頭調整指示
            draw_adjustment_arrow(frame, cva_angle, torso_inclination, l_ear_x, l_ear_y, l_shldr_x, l_shldr_y, l_hip_x,
                                  l_hip_y)
            # 限制每秒鐘只記錄一次姿勢狀態
            current_time = time.time()
            if last_log_time is None or current_time - last_log_time >= 1:  # 每秒鐘記錄一次
                log_posture(cva_angle, torso_inclination, posture_status)
                last_log_time = current_time

            cv2.putText(frame, f"CVA: {cva_angle:.1f} deg", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (127, 255, 0), 2)
            cv2.putText(frame, f"Torso: {torso_inclination:.1f} deg", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (127, 255, 0), 2)

        FRAME_WINDOW.image(frame)

    if cap and cap.isOpened():
        cap.release()

# Streamlit button to view historical data
st.sidebar.subheader("查看歷史數據")

# 讓使用者選擇開始和結束日期
start_date = st.sidebar.date_input("開始日期", min_value=datetime.date(2020, 1, 1), max_value=datetime.date.today(), value=datetime.date.today())
end_date = st.sidebar.date_input("結束日期", min_value=start_date, max_value=datetime.date.today(), value=datetime.date.today())

# 將日期轉換為 datetime 物件
start_time = datetime.datetime.combine(start_date, datetime.datetime.min.time())
end_time = datetime.datetime.combine(end_date, datetime.datetime.max.time())

# 讀取音效觸發事件資料
conn = sqlite3.connect("posture.db")
query_audio = "SELECT * FROM audio_event_data"
audio_data = pd.read_sql_query(query_audio, conn)

query_posture = "SELECT * FROM posture_data"
posture_data = pd.read_sql_query(query_posture, conn)
conn.close()

# 時間戳處理與篩選
audio_data["timestamp"] = pd.to_datetime(audio_data["timestamp"])
posture_data["timestamp"] = pd.to_datetime(posture_data["timestamp"])

# 根據使用者選擇的日期範圍過濾數據
audio_filtered_data = audio_data[(audio_data["timestamp"] >= start_time) & (audio_data["timestamp"] <= end_time)]
posture_filtered_data = posture_data[(posture_data["timestamp"] >= start_time) & (posture_data["timestamp"] <= end_time)]

# 用戶選擇顯示方式
display_mode = st.sidebar.selectbox("選擇顯示方式", ("長條圖", "數據表"))

# 根據用戶選擇顯示長條圖或數據表
if display_mode == "長條圖":
    # 計算每小時內音效觸發的次數
    audio_filtered_data["hour"] = audio_filtered_data["timestamp"].dt.strftime('%Y-%m-%d %H')
    audio_event_count = audio_filtered_data.groupby('hour').size()

    # 顯示長條圖
    fig, ax = plt.subplots()

    # 繪製長條圖
    ax.bar(audio_event_count.index, audio_event_count.values, color='r')

    # 設置 x 軸和 y 軸的標籤
    ax.set_xlabel("Y-M-D H")  # X 軸標籤
    ax.set_ylabel("Bad Posture Count")  # Y 軸標籤

    # 旋轉 x 軸的標籤以便更易讀
    plt.xticks(rotation=45)

    # 設置標題
    ax.set_title(f"({start_date} ~ {end_date})")

    # 顯示圖表
    st.pyplot(fig)

elif display_mode == "數據表":
    # 顯示姿勢數據的數據表，保留所有欄位
    st.write("姿勢數據表：")
    st.dataframe(posture_filtered_data)