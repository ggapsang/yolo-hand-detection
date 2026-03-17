import argparse
import cv2

from yolo import YOLODarknet, YOLOv11

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True, help='Path to video file')
ap.add_argument('-n', '--network', default="v11", choices=["v11", "normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-s', '--size', default=640, help='Size for yolo') # v11: 640, normal: 416, tiny: 416, prn: 160, v4-tiny: 416
ap.add_argument('-c', '--confidence', default=0.5, help='Confidence for yolo') # normal: 0.2, tiny: 0.1, prn: 0.1, v4-tiny: 0.05
ap.add_argument('-nh', '--hands', default=-1, help='Total number of hands to be detected per frame (-1 for all)') # 프레임 당 최대 감지할 손의 수 (기본값 -1은 모두 감지)
ap.add_argument('-o', '--output', default=None, help='Path to save output video (e.g. output.mp4). Default: no save')
args = ap.parse_args()

if args.network == "v11":
    print("loading yolov11 onnx...")
    yolo = YOLOv11("models/hand_detect_yolov11.onnx", ["hand"])
elif args.network == "normal":
    print("loading yolo...")
    yolo = YOLODarknet("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLODarknet("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny...")
    yolo = YOLODarknet("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLODarknet("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("opening video file...")
vc = cv2.VideoCapture(args.video)

if not vc.isOpened():
    print("Error: cannot open video file.")
    exit()

cv2.namedWindow("preview")

# 녹화 설정
writer = None
if args.output:
    fps = vc.get(cv2.CAP_PROP_FPS)
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    print(f"recording to {args.output}...")

while True:
    rval, frame = vc.read()

    if not rval:
        print("video ended.")
        break

    width, height, inference_time, results = yolo.inference(frame)

    # FPS 표시
    cv2.putText(frame, f'{round(1/inference_time, 2)} FPS', (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 신뢰도 기준 정렬
    results.sort(key=lambda x: x[2])

    hand_count = len(results)
    if args.hands != -1:
        hand_count = int(args.hands)

    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection

        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    cv2.imshow("preview", frame)

    if writer:
        writer.write(frame)

    key = cv2.waitKey(20)
    if key == 27:  # ESC
        break

cv2.destroyWindow("preview")
vc.release()
if writer:
    writer.release()
    print(f"saved to {args.output}")
