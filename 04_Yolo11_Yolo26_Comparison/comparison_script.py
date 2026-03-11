import cv2
import time
import os

from ultralytics import YOLO
from ultralytics.utils.plotting import colors

def draw_text_with_bg (img, text, pos, font_scale=0.6,
                      bg_color=(0, 0, 0), fg_color=(104, 31, 17),
                      padding=15, thickness=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    rw, rh = tw + 2 * padding, th + baseline + 2 * padding
    x1, y1 = pos
    x2, y2 = x1 + rw, y1 + rh
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    text_x = x1 + (rw - tw) // 2
    text_y = y1 + (rh + th) // 2
    cv2.putText(img, text, (text_x, text_y), font,
                font_scale, fg_color, thickness, cv2.LINE_AA)

def draw_cool_bbox (img, box, label="", cls_id=0):
    x1, y1, x2, y2 = map(int, box)
    color = colors(11 if cls_id == 2 else cls_id, True)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    if label:
        scale, thick = 1.4, 2
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        pad = 8
        lx1, ly1 = x1, y1 - (th + base + 2 * pad)
        lx2, ly2 = x1 + tw + 2 * pad, y1
        if ly1 < 0:
            ly1 = y1
            ly2 = y1 + th + base + 2 * pad

        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
        tx = lx1 + (lx2 - lx1 - tw) // 2
        ty = ly1 + (ly2 - ly1 + th) // 2
        cv2.putText(img, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

def run_yolo (source, model_path):

    display_scale = 0.5
    device = "cpu"

    model = YOLO (model_path, task="detect")
    names = model.names

    # -------- MODEL NAME EXTRACTION -------- #
    model_file = os.path.basename(model_path).lower()

    if "11" in model_file:
        model_label = "YOLO11"
    elif "26" in model_file:
        model_label = "YOLO26"
    else:
        model_label = model_file

    cap = cv2.VideoCapture(r"resources/" + source)
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * display_scale)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * display_scale)

    # Example: "output_yolo26n.onnx_video1.mp4"
    output_path = f"out_{model_path}_{source}"
    # Replace the first dot.
    output_path = output_path.replace(".", "_", count=1) 

    video_writer = cv2.VideoWriter(
        # "yolo-output.mp4",
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_input,
        (w, h)
    )

    avg_fps = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        start = time.time()
        results = model.predict(frame, device=device, conf=0.25, verbose=False)[0]
        process_time = time.time() - start

        boxes = results.boxes.xyxy.tolist()
        clss = results.boxes.cls.tolist()

        for box, cls in zip(boxes, clss):
            cls = int(cls)
            draw_cool_bbox(frame, box, label=names[cls], cls_id=cls)

        avg_fps.append(1.0 / process_time if process_time > 0 else 0)
        fps = sum(avg_fps) / len(avg_fps) # list

        draw_text_with_bg(frame, f'FPS: {fps:.2f}', (25, 45),
                          bg_color=(104, 31, 17), fg_color=(255, 255, 255), font_scale=1.7)

        draw_text_with_bg(frame, f'Time: {process_time*1000:.0f}ms', (360, 45),
                          bg_color=(104, 0, 123), fg_color=(255, 255, 255), font_scale=1.5)

        draw_text_with_bg(frame, f'Model: {model_label}', (700, 45),
                          bg_color=(0, 140, 255), fg_color=(255, 255, 255), font_scale=1.5)

        display_frame = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)

        # cv2.imshow("Ultralytics YOLO", display_frame)
        
        video_writer.write(display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def main ():
    for source in ["video1.mp4", "video2.mp4"]:
        for model in ["yolo11n.onnx", "yolo26n.onnx"]:
            print (f"Running {model} on {source}...")
            run_yolo (source, model)

if __name__ == "__main__":
    main ()

