from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np


# Load YOLO once when the backend starts
model = YOLO("yolo11n.pt")

# Create annotators once
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


def get_team_from_jersey(frame, box):
    """
    Estimate team from the player's jersey.

    Current video:
    Yellow Team = yellow jersey
    White Team = white jersey

    Returns:
        "Yellow Team"
        "White Team"
        "Unknown"
    """

    x1, y1, x2, y2 = [int(value) for value in box]

    # Keep coordinates inside the image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return "Unknown"

    # Focus on the torso where the jersey is located
    height = y2 - y1

    torso_y1 = y1 + int(height * 0.20)
    torso_y2 = y1 + int(height * 0.60)

    torso = frame[torso_y1:torso_y2, x1:x2]

    if torso.size == 0:
        return "Unknown"

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # ------------------------------------------------
    # YELLOW DETECTION
    # ------------------------------------------------

    yellow_lower = np.array([15, 70, 70])
    yellow_upper = np.array([40, 255, 255])

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    yellow_ratio = (
        np.count_nonzero(yellow_mask)
        / yellow_mask.size
    )

    # ------------------------------------------------
    # WHITE DETECTION
    # ------------------------------------------------

    white_lower = np.array([0, 0, 160])
    white_upper = np.array([180, 65, 255])

    white_mask = cv2.inRange(
        hsv,
        white_lower,
        white_upper
    )

    white_ratio = (
        np.count_nonzero(white_mask)
        / white_mask.size
    )

    # ------------------------------------------------
    # DARK / BLACK DETECTION
    # ------------------------------------------------

    dark_lower = np.array([0, 0, 0])
    dark_upper = np.array([180, 80, 90])

    dark_mask = cv2.inRange(
        hsv,
        dark_lower,
        dark_upper
    )

    dark_ratio = (
        np.count_nonzero(dark_mask)
        / dark_mask.size
    )

    # ------------------------------------------------
    # TEAM DECISION
    # ------------------------------------------------

    # Yellow jerseys
    if (
        yellow_ratio > 0.10
        and yellow_ratio > white_ratio
    ):
        return "Yellow Team"

    # White jerseys
    #
    # Require a fairly large amount of white AND
    # prevent strongly black/white people from being
    # classified as white-team players.
    if (
        white_ratio > 0.25
        and white_ratio > yellow_ratio * 1.3
        and dark_ratio < 0.25
    ):
        return "White Team"

    return "Unknown"


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

    if not out.isOpened():
        cap.release()
        raise ValueError("Could not create output video")

    # IMPORTANT:
    # Create a fresh tracker for every video.
    tracker = sv.ByteTrack()

    # Store team votes for each tracked player.
    #
    # Example:
    # Player 17:
    #     Yellow Team = 25
    #     White Team = 2
    #
    # This makes the final team assignment more stable.
    team_votes = {}

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ------------------------------------------------
        # YOLO
        # ------------------------------------------------

        results = model(
            frame,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(
            results
        )

        # Keep only people
        person_mask = detections.class_id == 0

        detections = detections[person_mask]

        # ------------------------------------------------
        # BYTE TRACK
        # ------------------------------------------------

        detections = tracker.update_with_detections(
            detections
        )

        labels = []

        if detections.tracker_id is not None:

            for tracker_id, box in zip(
                detections.tracker_id,
                detections.xyxy
            ):

                tracker_id = int(tracker_id)

                # Create vote storage
                if tracker_id not in team_votes:

                    team_votes[tracker_id] = {
                        "Yellow Team": 0,
                        "White Team": 0
                    }

                # Look at jersey color
                team = get_team_from_jersey(
                    frame,
                    box
                )

                # Add a vote
                if team in team_votes[tracker_id]:
                    team_votes[tracker_id][team] += 1

                yellow_votes = team_votes[tracker_id][
                    "Yellow Team"
                ]

                white_votes = team_votes[tracker_id][
                    "White Team"
                ]

                total_votes = (
                    yellow_votes
                    + white_votes
                )

                # Don't make a confident decision
                # until we've seen the player several times.
                if total_votes < 5:

                    stable_team = "Unknown"

                elif yellow_votes > white_votes:

                    stable_team = "Yellow Team"

                elif white_votes > yellow_votes:

                    stable_team = "White Team"

                else:

                    stable_team = "Unknown"

                labels.append(
                    f"{stable_team} | Player {tracker_id}"
                )

        # ------------------------------------------------
        # DRAW BOXES
        # ------------------------------------------------

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections
        )

        # ------------------------------------------------
        # DRAW LABELS
        # ------------------------------------------------

        if len(labels) > 0:

            frame = label_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

        # ------------------------------------------------
        # WRITE FRAME
        # ------------------------------------------------

        out.write(frame)

        # Progress every 30 frames
        if frame_number % 30 == 0:

            print(
                f"Processed {frame_number} frames | "
                f"Detected {len(detections)} people"
            )

    cap.release()
    out.release()

    print(
        f"Analysis complete. "
        f"Processed {frame_number} frames."
    )

    print(
        f"Output saved to: {output_path}"
    )

    return output_path
