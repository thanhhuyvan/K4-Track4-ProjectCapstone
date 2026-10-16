# Adaptive DMS

Hệ thống giám sát tài xế (Driver Monitoring System) chạy realtime trên CPU, đánh giá trên
bộ dữ liệu [DMD](https://dmd.vicomtech.org/). MediaPipe chạy **một lần mỗi frame** và dùng chung cho
nhánh gaze (BlazeGaze) và nhánh mắt/miệng; trạng thái tài xế được quyết định bằng các
**cửa sổ trượt chỉ dùng dữ liệu quá khứ (causal)**.

> Trạng thái: nguyên mẫu nghiên cứu. Gaze và tín hiệu mắt thô đã được đo trên DMD; các nhãn
> `microsleep`/`drowsy` trong demo là **luật chưa được kiểm chứng** (xem [Hạn chế](#hạn-chế)).

## Kiến trúc

```text
webcam / video
  └─ MediaPipe FaceLandmarker (478 điểm, blendshape, ma trận pose)   ← 1 lần / frame
       ├─ Gaze: dải mắt 128×512 → BlazeGaze (đóng băng) → hiệu chỉnh theo người lái → 1/9 vùng nhìn
       ├─ Mắt:  crop affine 16×16 mỗi mắt, EAR, blendshape eyeBlink, góc quay đầu
       └─ Miệng: blendshape jawOpen
            └─ cửa sổ trượt causal: PERCLOS 30 s, độ dài lần nhắm, thời gian ngáp,
               tỉ lệ nhìn khỏi đường 2 s → trạng thái: alert / distracted / yawning / drowsy / microsleep
```

## Demo

Cần Python 3.11–3.12 và webcam.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,train]"
```

Tải hai file vào `weights/` (không nằm trong Git) từ
[WebEyeTrack](https://github.com/RedForestAI/WebEyeTrack) (`python/webeyetrack/model_weights/`):
`blazegaze_mpiifacegaze.keras` và `face_landmarker_v2_with_blendshapes.task`.

```powershell
.\run_demo.ps1                 # webcam; lần đầu hiệu chỉnh gaze ~80 s (nhìn theo từng mục tiêu)
.\run_demo.ps1 -NoGaze         # chỉ mắt / ngáp / hướng đầu
.\run_demo.ps1 -Source clip.mp4 -NoGaze -Save out.mp4 -NoDisplay
```

Phím: `q` thoát, `space` tạm dừng, `r` reset cửa sổ, `w` tự dò lại vô lăng (với `--hands`). Hiệu chỉnh gaze lưu ở
`weights/gaze_profile.npz` (dữ liệu sinh trắc học của bạn, đã được `.gitignore` chặn).
**Tay trên vô lăng** (`-Hands`): cần thêm `weights/hand_landmarker.task` (tải từ
[MediaPipe](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)).
Dựng một vô lăng giả (đĩa giấy / vòng bìa) **đứng yên** trước webcam, rồi click vào tâm, sau đó vào
vành. Với video DMD camera tay: `--driver-left-on-image-left --wheel-radius-range 0.28 0.40` để tự dò
vô lăng. `--log-csv run.csv` rồi `python scripts/plot_timeline.py run.csv` vẽ timeline chung
(trạng thái tài xế + trạng thái tay).

Nếu có `weights/eye_cnn_16.pt` (xem bên dưới), xác suất của CNN mắt được hiển thị cạnh EAR và
blendshape; thêm `--eye-signal cnn` để CNN quyết định mắt nhắm/mở.

## Kết quả đã đo

Số liệu nằm trong [results/](results/); mọi con số dưới đây đều lấy từ các file đó.

**Hiệu chỉnh gaze theo người lái** (4 driver DMD `s6`; K frame mỗi vùng ở đầu phiên, đánh giá trên
frame sau của chính người đó; `spread` = trải đều, `burst` = liên tiếp như hiệu chỉnh có hướng dẫn):

| Cách | Zone-9 F1 | Distracted F1 | Sai số PoG |
|---|---:|---:|---:|
| BlazeGaze zero-shot | 0,081 | 0,573 | 0,421 |
| hiệu chỉnh `spread` K=5 | 0,565 | 0,781 | 0,234 |
| hiệu chỉnh `burst` K=20 | 0,467 | 0,711 | 0,247 |

**Mắt nhắm** (nhãn `blinking` của DMD):

| Thí nghiệm | Kết quả |
|---|---|
| CNN 16×16 (≈48 nghìn tham số) trên subject lạ `gB/7` | AUC 0,83, kém blendshape (0,93) và EAR (0,91) |
| LOSO 4 video, cửa sổ 1 s: GRU / window-logistic / attention | F1 0,64 / 0,59 / 0,56; AUC ≈ 0,87 cho tất cả |

Attention nhẹ **không** đạt tiêu chí đã đặt trước (AUC cao hơn ≥ 0,02 và thắng ≥ 3/4 fold), nên
chưa có bằng chứng nó hữu ích với dữ liệu này.

**Tay trên vô lăng** (`--hands`, [results/hands_lodo.json](results/hands_lodo.json)): chính
code chạy trong demo (`src/adaptive_dms/hands.py`), chạy trên camera tay của DMD, đánh giá
leave-one-driver-out (chỉnh tham số trên 3 người, đo trên người thứ 4; 12,8 phút, 19 lần buông tay):

| Chỉ số | Kết quả |
|---|---|
| Frame buông tay bắt được | 92% trung bình, 86% ở người tệ nhất |
| Lần buông tay bắt được | 18/19 |
| Độ trễ báo động (trung vị) | 0,27 s, chỉ dùng frame quá khứ |
| Báo động sai | 31 lần, tổng 68 s |
| Macro-F1 4 trạng thái (cả hai / trái / phải / buông) | 0,54; chỉ 0,28 ở `gA/1` |

Cách làm: vô lăng đứng yên nên trung vị (median) của ~150 frame xóa tay đang chuyển động, rồi
Hough tìm vành vô lăng — không cần nhãn. Một bàn tay "trên vô lăng" khi tâm 21 điểm MediaPipe nằm
trong 1,2 bán kính vô lăng; trạng thái mới phải giữ ≥ 0,17 s mới được báo. Trụ cột Track 4:
**Hình học** (vô lăng → hệ tọa độ theo bán kính), **Bất định** (ngưỡng điểm tin cậy, debounce),
**Thời gian** (causal, đo độ trễ), **Bằng chứng** (đo trên người lạ, ca lỗi bên dưới).

Ca lỗi đã phân tích: tay phải đặt trong lòng *ngay dưới* vành (≈ 1,05 bán kính) bị tính là đang
cầm; khi tay phải vắt qua vô lăng, ống tay áo che bàn tay, MediaPipe không thấy tay nên báo buông.
Phân biệt trái/phải yếu vì MediaPipe chỉ thấy tay phải ở 21% frame của `gA/1`. So sánh: mô hình
XGBoost hai camera (không chạy được trên webcam) bắt 96% frame buông tay, macro-F1 0,84.

**Kích thước** (không tính MediaPipe): ≈ 281 nghìn tham số, ≈ 3,1 GFLOP/s ở 30 FPS mắt + 10 FPS
gaze; BlazeGaze chiếm ≈ 95%. Demo chạy ≈ 30 FPS trên CPU 12 luồng.

## Hạn chế

- **DMD `s6` chỉ có nhãn chớp mắt và vùng nhìn.** Không có nhãn `drowsy`, `microsleep`, `yawning`
  nên chưa thể huấn luyện hay đo bộ phân loại 5 trạng thái. Các ngưỡng trong
  `src/adaptive_dms/realtime.py` đặt tay.
- **Nhìn xuống bị nhầm thành nhắm mắt.** Trên `gA/5`, khi nhìn vô lăng/màn hình giữa, luật mắt
  báo "nhắm" tới 99% và 49% frame trong khi nhãn thực là 8% và 0%. Một lọc theo góc quay đầu đã
  giảm lỗi này ở `gB/7` nhưng chưa đủ ở `gA/1`, `gA/5`.
- Chỉ 4 subject; subject `gB/6` có rất ít nhãn chớp (139 frame so với ≈ 1.000 ở hai người còn lại).
- Hiệu chỉnh gaze được đo trên cùng người lái ở các frame sau, không phải người lạ.
- Trên webcam để bàn, 9 vùng nhìn là quy ước của bạn, không phải cabin xe thật.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `src/adaptive_dms/data/dmd.py` | Đọc annotation OpenLABEL của DMD |
| `src/adaptive_dms/dms/` | Cổng PyTorch của BlazeGaze (nạp `.keras`), CNN mắt 16×16, model/policy cũ |
| `src/adaptive_dms/gaze_calibration.py` | Hiệu chỉnh gaze: affine + fine-tune đầu MLP |
| `src/adaptive_dms/realtime.py` | Crop 16×16, cửa sổ trượt, `DriverMonitor` |
| `src/adaptive_dms/hands.py` | Dò vô lăng, luật tay-trên-vành, `HandsMonitor` (sự kiện `hands_off`) |
| `scripts/eval_hands_dmd.py`, `plot_timeline.py` | Đánh giá nhánh tay trên DMD; timeline chung |
| `scripts/demo_realtime.py`, `calibrate_gaze.py` | Demo và hiệu chỉnh có hướng dẫn |
| `scripts/build_eye_crop_dataset.py`, `train_eye_crop.py` | Dựng dataset crop và train CNN mắt |
| `scripts/fuse_eye_signals.py`, `train_eye_temporal.py` | So sánh gộp tín hiệu và mô hình cửa sổ |
| `scripts/cache_dmd_blazegaze.py`, `eval_gaze_calibration.py`, `train_loso_dmd_blazegaze.py` | Cache BlazeGaze và đánh giá gaze |
| `scripts/cache_dmd_mediapipe.py`, `train_dmd_baseline.py`, `evaluate_dmd_events.py` | Nhánh ngáp/mắt cũ trên DMD `s5` (cần dữ liệu `s5`, chưa tái lập trong repo này) |
| `tests/` | 31 test (`pytest`) |
| `MASTERPLAN.md` | Kế hoạch dự án |

## Tái lập

Đặt DMD vào `data/dmd/` (xin quyền từ Vicomtech), rồi:

```powershell
python scripts/cache_dmd_blazegaze.py --gaze-valid-only
python scripts/eval_gaze_calibration.py
python scripts/build_eye_crop_dataset.py --output artifacts/eye_feat_16
python scripts/train_eye_crop.py --data artifacts/eye_feat_16    # lưu weights/eye_cnn_16.pt
python scripts/train_eye_temporal.py --data artifacts/eye_feat_16
pytest
```

## Dữ liệu và giấy phép

Video DMD, cache `.npz`, hồ sơ hiệu chỉnh và checkpoint là dữ liệu sinh trắc học nên không bao giờ
được commit. Kiến trúc và trọng số BlazeGaze lấy từ WebEyeTrack (MIT, © 2025 Davalos, Zhang,
Goodwin, Biswas).
