import datetime
import time
import cv2
from ultralytics import YOLO, solutions

def test_video(model: str, video: str, output_name: str):
    model = YOLO(model)
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    line_points = [
    (int(w * 0.05), int(h * 0.05)),  # Top-left
    (int(w * 0.95), int(h * 0.05)),  # Top-right
    (int(w * 0.95), int(h * 0.95)),  # Bottom-right
    (int(w * 0.05), int(h * 0.95))   # Bottom-left
    ]
    
    video_writer = cv2.VideoWriter(f"{output_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counter = solutions.ObjectCounter(
        view_img=False,  
        reg_pts=line_points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )
    max_frames = int(fps * 100)
    frame_count = 0
    start_time = datetime.datetime.now()

    while cap.isOpened() and frame_count < max_frames:
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        tracks = model.track(im0, persist=True, show=False)
        im0 = counter.start_counting(im0, tracks)
        current_time = (start_time + datetime.timedelta(seconds=frame_count / fps)).strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(im0, f"Timestamp: {current_time}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        video_writer.write(im0)
        frame_count += 1

    cap.release()
    video_writer.release()


if __name__ == "__main__":
    curr_time = time.time()
    test_video("models/prenabhismallhyp125.pt", "videos/SBI_Bnk_JN_FIX_1_000.mp4", "processed_output_sbi_hyp_small")
    process_time = time.time() - curr_time
    print(f"Processing time: {process_time:.2f} seconds")