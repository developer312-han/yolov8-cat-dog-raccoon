# YOLOv8n fine-tuning: cat / dog / raccoon

사전학습된 YOLOv8n을 고양이(cat), 강아지(dog), 라쿤(raccoon) 3개 부류 데이터로 fine-tuning하고,
영상에서 객체를 감지·추적한 결과를 정리한 저장소입니다. (컴퓨터 비전 과제)

## 데이터

- 출처: Roboflow Universe, [Animal Real Data V1](https://universe.roboflow.com/saksorn-ruangtanusak/animal-real-data-v1) (version 2), 작성자 Saksorn ruangtanusak
- 라이선스: CC BY 4.0
- 이미지 556장, Object Detection(박스 라벨), 클래스 3개
- 원본의 valid/test에는 raccoon 이미지만 있어서, 전체를 합친 뒤 부류별로 고르게 약 70/20/10으로 다시 나누었습니다 (seed 0).

| split | 이미지 | cat | dog | raccoon |
| --- | --- | --- | --- | --- |
| train | 387 | 137 | 150 | 115 |
| valid | 112 | 41 | 44 | 36 |
| test | 57 | 19 | 20 | 18 |

(부류 열은 박스 수)

## 학습 설정

| 항목 | 값 |
| --- | --- |
| 모델 | yolov8n.pt (COCO 사전학습) |
| 프레임워크 | Ultralytics 8.4.218 |
| 환경 | Google Colab, Tesla T4 |
| epochs / patience | 50 / 20 |
| imgsz / batch | 640 / 16 |
| optimizer | auto (AdamW, lr 0.001429) |
| 학습 시간 | 약 5분 |

## 성능 (valid 112장, best.pt)

| 부류 | Precision | Recall | AP50 | AP50-95 |
| --- | --- | --- | --- | --- |
| cat | 0.881 | 0.780 | 0.803 | 0.585 |
| dog | 0.973 | 0.815 | 0.953 | 0.710 |
| raccoon | 0.958 | 0.833 | 0.917 | 0.615 |
| 전체 | 0.937 | 0.810 | 0.891 | 0.636 |

학습 곡선은 `results.png`, 혼동 행렬은 `confusion_matrix.png`, PR 곡선은 `BoxPR_curve.png`를 참고하세요.

## 영상 결과

- `test_result.mp4`: 20초 추적 결과에서 구간을 발췌·편집한 약 5.8초 영상입니다. 연속 구간이 아니므로 화면의 추적 ID와 왼쪽 위 부류별 카운터는 20초 전체 처리의 누적값입니다.
- `test_full.mp4`: 20초 전체 구간의 감지·추적 결과입니다.
- 추적은 ByteTrack(`bytetrack.yaml`, `conf=0.25`, `iou=0.5`)을 사용했고, 박스 중심점의 최근 30프레임 궤적과 부류별 고유 ID 수를 영상 위에 그렸습니다.

## 한계

- 영상에서 소파 위 회색 고양이가 확인한 프레임 전부에서 dog으로 검출되었고, 일부 프레임에서 라쿤이 cat, 강아지가 cat으로 나오는 오분류가 있었습니다.
- valid 성능에서는 cat→dog 오분류가 41개 중 3개로 작아, 영상 환경이 학습 데이터와 달랐던 것이 원인일 가능성이 있습니다(추정).
- ID가 끊기거나 다른 객체에 재사용되는 경우가 있어, 영상의 부류별 고유 ID 수는 실제 개체 수보다 큽니다.

## 파일

- `best.pt`: 학습한 모델 가중치
- `yolo_finetune.py`: 학습·평가·영상 생성 스크립트
- `results.png`, `confusion_matrix.png`, `BBoxPR_curve.png`: 학습 결과 그래프
- `test_result.mp4`, `test_full.mp4`: 감지·추적 영상

## 사용 방법

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.track("your_video.mp4", tracker="bytetrack.yaml", conf=0.25, iou=0.5, save=True)
```

## 라이선스와 출처

데이터셋은 CC BY 4.0(작성자 Saksorn ruangtanusak)이며, 모델은 Ultralytics YOLO(AGPL-3.0)를 기반으로 합니다.
