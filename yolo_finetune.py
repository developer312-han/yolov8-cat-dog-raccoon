"""
YOLO fine-tuning 과제 (Colab 용) — 3개 부류 커스텀 데이터 학습 + 5초 tracking 영상 생성

사용 순서 (Colab, 런타임 > T4 GPU):
  !pip install -q ultralytics
  1) Roboflow 에서 YOLOv8 포맷으로 export 한 zip 을 /content/dataset 에 풀기
     (data.yaml, train/, valid/, test/ 구조)
  2) 이 파일의 DATA_YAML, VIDEO_SRC 만 수정
  3) 셀에서 단계별로 실행:  train() -> evaluate() -> make_track_video()
"""
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

# ------------------------------------------------------------------ 설정
DATA_YAML = "/content/dataset/data.yaml"      # Roboflow export 의 data.yaml
BASE_WEIGHTS = "yolov8n.pt"                    # 슬라이드 기준 YOLOv8 (yolo11n.pt 도 가능)
PROJECT, RUN = "runs_hw", "custom3"
VIDEO_SRC = "/content/test_video.mp4"          # 추적 데모용 원본 영상 (3개 부류 객체가 나오는 영상)
OUT_VIDEO = "/content/track_result.mp4"
CLIP_SECONDS = 5


# ------------------------------------------------------------------ 1. 학습
def train():
    model = YOLO(BASE_WEIGHTS)                 # COCO pretrained -> 전이학습
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        patience=20,                           # early stopping
        project=PROJECT,
        name=RUN,
        # freeze=10,                           # 백본 앞 10개 레이어 고정하고 싶을 때 (데이터가 매우 적을 때)
        # 헤드의 nc(클래스 수)는 data.yaml 의 nc/names 에 맞춰 자동 재구성됨
    )
    return Path(PROJECT) / RUN


# ------------------------------------------------------------------ 2. 성능 평가
def evaluate():
    best = Path(PROJECT) / RUN / "weights" / "best.pt"
    model = YOLO(str(best))
    m = model.val(data=DATA_YAML, split="val")   # test 셋이 있으면 split="test"
    print(f"mAP50     : {m.box.map50:.4f}")
    print(f"mAP50-95  : {m.box.map:.4f}")
    print(f"precision : {m.box.mp:.4f}")
    print(f"recall    : {m.box.mr:.4f}")
    for i, name in model.names.items():          # 부류별 AP
        print(f"  {name:15s} AP50={m.box.ap50[i]:.4f}  AP50-95={m.box.ap[i]:.4f}")

    # 에폭별 loss / mAP 곡선 값 (보고서 그래프용): results.csv, results.png 는 학습 폴더에 자동 저장됨
    with open(Path(PROJECT) / RUN / "results.csv") as f:
        rows = list(csv.DictReader(f))
    print("last epoch:", {k.strip(): v for k, v in rows[-1].items()})


# ------------------------------------------------------------------ 3. 5초 tracking 영상
def make_track_video():
    best = Path(PROJECT) / RUN / "weights" / "best.pt"
    model = YOLO(str(best))

    cap = cv2.VideoCapture(VIDEO_SRC)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    max_frames = int(fps * CLIP_SECONDS)
    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trails = {}                                  # track_id -> [(cx, cy), ...] 궤적
    seen = {}                                    # 부류별 고유 track id (counting)
    n = 0
    while cap.isOpened() and n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        # persist=True: 프레임 간 track ID 유지 (프레임을 직접 넘길 때 필수)
        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        conf=0.3, iou=0.5, verbose=False)[0]
        vis = r.plot()                           # bbox + 클래스 + track id 그려진 프레임
        if r.boxes is not None and r.boxes.id is not None:
            xywh = r.boxes.xywh.cpu().tolist()
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            for (cx, cy, _, _), tid, c in zip(xywh, ids, clss):
                trails.setdefault(tid, []).append((int(cx), int(cy)))
                seen.setdefault(model.names[c], set()).add(tid)
                pts = trails[tid][-30:]
                for a, b in zip(pts, pts[1:]):
                    cv2.line(vis, a, b, (0, 255, 255), 2)   # 중심점 궤적
        txt = " | ".join(f"{k}:{len(v)}" for k, v in seen.items())
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        writer.write(vis)
        n += 1
    cap.release()
    writer.release()
    print(f"saved {OUT_VIDEO} ({n} frames, {n / fps:.1f}s)")
    # Colab 에서 다운로드: from google.colab import files; files.download(OUT_VIDEO)


if __name__ == "__main__":
    train()
    evaluate()
    make_track_video()
