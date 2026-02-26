"""
Simple Gait Feature Extraction using MediaPipe Pose
Author: [Your Name]
Course: [Course Name]
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def create_output_directory():
    """Ensure output directory exists."""
    output_dir = "gait_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def calculate_angle(a, b, c):
    """
    Compute joint angle at point b using three points (a, b, c).
    Angle is calculated using vector dot product formula.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


# ------------------------------------------------------------
# Pose Extraction
# ------------------------------------------------------------

def extract_keypoints(video_path, output_dir):
    """
    Extract required joint keypoints frame-by-frame.
    Save as CSV and return dataframe.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)

    data = []
    frame_count = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_video = cv2.VideoWriter(
        os.path.join(output_dir, "skeleton_overlay.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Required joints
            joints = {
                "l_hip": landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                "r_hip": landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                "l_knee": landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                "r_knee": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                "l_ankle": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                "r_ankle": landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            }

            row = [frame_count]
            for joint in joints.values():
                row.extend([joint.x, joint.y])

            data.append(row)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out_video.write(frame)
        frame_count += 1

    cap.release()
    out_video.release()

    columns = ["frame",
               "l_hip_x","l_hip_y",
               "r_hip_x","r_hip_y",
               "l_knee_x","l_knee_y",
               "r_knee_x","r_knee_y",
               "l_ankle_x","l_ankle_y",
               "r_ankle_x","r_ankle_y"]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(output_dir, "extracted_keypoints.csv"), index=False)

    return df, fps


# ------------------------------------------------------------
# Feature Computation
# ------------------------------------------------------------

def compute_joint_angles(df):
    """Compute knee and hip angles frame-by-frame."""

    l_knee_angles = []
    r_knee_angles = []
    hip_angles = []

    for _, row in df.iterrows():

        l_hip = [row.l_hip_x, row.l_hip_y]
        l_knee = [row.l_knee_x, row.l_knee_y]
        l_ankle = [row.l_ankle_x, row.l_ankle_y]

        r_hip = [row.r_hip_x, row.r_hip_y]
        r_knee = [row.r_knee_x, row.r_knee_y]
        r_ankle = [row.r_ankle_x, row.r_ankle_y]

        l_knee_angles.append(calculate_angle(l_hip, l_knee, l_ankle))
        r_knee_angles.append(calculate_angle(r_hip, r_knee, r_ankle))

        hip_angles.append(calculate_angle(l_knee, l_hip, r_hip))

    return np.array(l_knee_angles), np.array(r_knee_angles), np.array(hip_angles)


def compute_temporal_features(df, fps):
    """Estimate step duration and cadence using ankle vertical motion."""

    l_ankle_y = df["l_ankle_y"].values
    peaks, _ = find_peaks(-l_ankle_y, distance=10)

    step_times = np.diff(peaks) / fps
    avg_step_duration = np.mean(step_times)

    cadence = 60 / avg_step_duration

    avg_ankle_disp = np.mean(np.abs(np.diff(l_ankle_y)))

    return avg_step_duration, cadence, avg_ankle_disp


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_angles(time, l_knee, r_knee, output_dir):
    plt.figure()
    plt.plot(time, l_knee, label="Left Knee")
    plt.plot(time, r_knee, label="Right Knee")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.legend()
    plt.title("Knee Angle vs Time")
    plt.savefig(os.path.join(output_dir, "knee_angle_plot.png"))
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    video_path = "walkingVideo.mp4"
    output_dir = create_output_directory()

    df, fps = extract_keypoints(video_path, output_dir)

    l_knee, r_knee, hip = compute_joint_angles(df)

    time = np.arange(len(l_knee)) / fps

    plot_angles(time, l_knee, r_knee, output_dir)

    avg_step_duration, cadence, avg_ankle_disp = compute_temporal_features(df, fps)

    feature_vector = [
        np.mean(l_knee),
        np.var(l_knee),
        np.mean(r_knee),
        np.var(r_knee),
        np.mean(hip),
        avg_step_duration,
        cadence,
        avg_ankle_disp
    ]

    print("\nFinal Gait Feature Vector:")
    print(feature_vector)

    np.savetxt(os.path.join(output_dir, "gait_feature_vector.txt"),
               feature_vector,
               fmt="%.4f")

if __name__ == "__main__":
    main()