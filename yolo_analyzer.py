from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")


def analyze_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Could not open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        out.write(annotated_frame)

    cap.release()
    out.release()

    return output_path