import argparse
import time
import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

MODEL_PATH = "yolo11n.pt"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input filename")
    parser.add_argument("-o", "--output", help="output filename")
    parser.add_argument("-c", "--confidence", type=int, choices = range(1, 100), metavar="1-99", 
                            default=50, help="confidence threshold in %% (default: 50)" )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v per-frame YOLO processing")
    return parser.parse_args()

def open_source(path: Path) -> cv2.VideoCapture:
    if path.is_dir():
        cap = cv2.VideoCapture(str( path / "img_%05d.bmp"))
    else:
        cap = cv2.VideoCapture(str(path))

    assert cap.isOpened(), "Error reading input file"
    return cap

def make_writer(path, cap) -> cv2.VideoWriter:
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return video_writer

def detect(model, frame, confidence, verbose) -> tuple[np.ndarray, int]:
    results = model(frame, conf=confidence, classes=[0], verbose=(verbose >= 1)) 
    annotated_frame = results[0].plot(line_width=1)

    n_people = len(results[0].boxes)

    return annotated_frame, n_people

def draw_overlay(annotated_frame, n_people, fps) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated_frame, f"Number of people detected: {n_people}", (2, 10), font, 0.22, (0, 0, 0), 2, cv2.LINE_AA )
    cv2.putText(annotated_frame, f"Processing FPS: {fps}", (2, 17), font, 0.22, (0, 0, 0), 2, cv2.LINE_AA )
    
def print_report(total_frames, detected_frames, elapsed_time, detection_time):
    if total_frames == 0:
        return
    print("---- RESULTS: ")
    print(f"Model name: {Path(MODEL_PATH).stem}")
    print(f"Total frames: {total_frames}\nDetected frames: {detected_frames}\nPercentage: {((detected_frames/total_frames)*100):.2f}%")

    print(f"Total elapsed time: {elapsed_time:.2f} s ({elapsed_time * 1000:.0f} ms)")
    print(f"Total detection time: {detection_time:.2f} s ({detection_time * 1000:.0f} ms)")
    
def release(cap, writer):
    cap.release()
    if writer is not None:
        writer.release()
    else:
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    confidence = float(args.confidence)/100

    model = YOLO(MODEL_PATH)
    cap = open_source(Path(args.input))
    writer = make_writer(args.output, cap) if args.output else None

    start = time.perf_counter()
    prev = start
    total_frames = detected_frames = 0
    detection_time = 0.0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        t0 = time.perf_counter()
        annotated_frame, n_people = detect(model, frame, confidence, args.verbose)
        detection_time += time.perf_counter() - t0 

        now = time.perf_counter()
        fps = int(1 / (now - prev))
        prev = now

        total_frames += 1
        if n_people >= 1:
            detected_frames += 1
    
        draw_overlay(annotated_frame, n_people, fps)

        if args.output:
            writer.write(annotated_frame)
        else:
            cv2.imshow("YOLO Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed_time = time.perf_counter() - start
    release(cap, writer)

    print_report(total_frames, detected_frames, elapsed_time, detection_time)


if __name__ == "__main__":
    main()

