import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from warnings import filterwarnings
filterwarnings('ignore')

# Загрузка кадров
def load_video(VIDEO_PATH, algorithm):

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(f"{algorithm}_output.mp4", fourcc, fps_video, (w, h))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Загружено кадров: {len(frames)}")
    return frames, out

def main(VIDEO_PATH, CONF_THRESHOLD = 0.5, algorithm = 'YOLOv8'):
    
    frames, out = load_video(VIDEO_PATH, algorithm)

    if algorithm == 'YOLOv8':
        print('Алгоритм - YOLOv8')
        yolo = YOLO("yolov8n.pt")

        start = time.time()
        detections = 0
        confidences = []

        for frame in frames:
            results = yolo.predict(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            confidences.extend(scores)
            detections += len(boxes)

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                label = f'Person: {score:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            out.write(frame)

        fps = len(frames) / (time.time() - start)


    elif algorithm == 'Faster R-CNN':
        print('Алгоритм - Faster R-CNN')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        frcnn = fasterrcnn_resnet50_fpn(pretrained=True)
        frcnn.to(device)
        frcnn.eval()

        start = time.time()
        detections = 0
        confidences = []
        for frame in frames:
            img_tensor = F.to_tensor(frame).to(device)

            with torch.no_grad():
                preds = frcnn([img_tensor])[0]

            boxes = preds["boxes"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()

            mask = (labels == 1) & (scores > CONF_THRESHOLD)
            boxes = boxes[mask]
            scores = scores[mask]
            confidences.extend(scores)

            detections += len(boxes)

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                label = f"Person: {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)

        fps = len(frames) / (time.time() - start)

    else:
        print('Такой реализации нет!')
        return -1

    out.release()
    print(f"{algorithm}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Среднее число детекций: {detections / len(frames):.2f}")

    plt.figure()
    plt.hist(confidences, bins=50, alpha=0.4, label=algorithm)
    plt.xlabel("Confidence")
    plt.ylabel("Количество детекций")
    plt.title("Распределение confidence")
    plt.legend()
    plt.grid()
    plt.savefig(f'./{algorithm}_distribution_confidence.png')
    plt.show()


if __name__ == '__main__':
    '''
    Args:
        VIDEO_PATH - путь до видео
        CONF_THRESHOLD - порог значения confidence
        algorithm - алгоритм ('YOLOv8' или 'Faster R-CNN')
    
    Return:
        None
    '''
    main(VIDEO_PATH='./crowd.mp4', CONF_THRESHOLD=0.6, algorithm='Faster R-CNN')

