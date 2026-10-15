# 단일 이미지·단일 z plane CITL

실행 파일은 `citl_closed_loop.py` 하나입니다. 저장소를 clone/pull하면 CITL 어댑터,
`citl_optics/`의 검증된 CSSNet·광학 실행 코드, `preprocess_hdr_aligned_camera.py`가
함께 준비됩니다. 저장소 루트에서 실행하고 설정 JSON에 실험 PC의 데이터 경로를 지정하세요.
Checkpoint, raw training data, LUT, phase/target 파일, hololith 장비 SDK는 별도로 준비합니다.
`citl_optics/PROVENANCE.json`과 `citl_camera_backend_source.json`에 포함한 원본과 SHA256을 기록했습니다.

## 설정

`citl_config.example.json`을 복사해서 `citl_config.local.json`으로 저장합니다.
현재 offline optimizer의 target, checkpoint, optics preset, target 방향,
z=6 mm, shape loss, Adam lr=0.1, 30 updates를 예제에 반영했습니다.
`null` 또는 `EDIT_ME` 항목과 다른 PC에서 달라진 경로를 수정하세요.

| 항목 | 의미 |
| --- | --- |
| `proxy.proxy_source_dir` | 기본값 `citl_optics`: 이 저장소에 포함한 검증된 CSSNet·광학 실행 코드. 다른 optical source tree를 사용하려면 해당 폴더 지정 |
| `proxy.checkpoint` | 해당 optical setup으로 학습한 CSSNet/shared_optical `best.pt` |
| `proxy.phase` | offline optimizer가 저장한 `phase_best_slm.npy` 등 시작할 phase 파일 |
| `proxy.target` | offline optimization에 사용한 같은 target |
| `reference_phase` | training의 `reference_captures` 촬영에 사용한 **고정 reference phase**의 integer `.npy` |
| `preprocessing.training_root` | `Exp300`, `Exp1000`, `Exp5000` 등이 있는 raw training 상위 폴더 |
| `preprocessing.preprocessing_script` | 기본값 `preprocess_hdr_aligned_camera.py`: 저장소에 포함한 원본 전처리 코드 |
| `preprocessing.report_path` | training 때 실제 사용한 `preprocessing_report.json`; 기본 위치이면 `null` |
| `preprocessing.lut_path` | 같은 camera response LUT (`.xlsx` 또는 원래 전처리가 지원하는 형식) |
| `hardware.hololith_root` | `hololith` 폴더를 포함한 `3DHL-Python-Control` 경로 |
| `hardware.config_path` | SLM 번호·Arduino COM 등 실제 장비의 `setting.yaml` |
| `hardware.camera_serial` | 여러 카메라가 연결되면 serial을 지정; `null`이면 첫 카메라 |
| `hardware.camera_gain` | training/LUT 측정과 동일한 sensor gain; `null`이면 현재 값 유지, auto gain은 비활성화 |

`reference_phase`는 calibration용 checkerboard phase와 다를 수 있습니다.
Report의 `anchor_stem`은 reference **촬영 파일** 이름이며 reference phase 파일
이름을 알려주지는 않습니다. Target을 reference로 사용하면 안 됩니다.

전형적인 데이터 구조는 다음과 같습니다. 프로그램은 원래 전처리 스크립트의
폴더 검색 및 파일 매칭 함수를 사용합니다.

```text
training_raw_root/
  Exp300/
    <training_pattern>.raw
    reference_captures/<training_pattern>.raw
    ...
  Exp1000/...
  Exp5000/...
  Aligned_camera_v3/
    preprocessing_report.json
    z_0/...
```

다른 위치에 결과를 저장했다면 `report_path`를 명시하세요. Offset이 있는 데이터는
`Exp300_z+1mm` 등 기존 규칙으로 검색합니다. `proxy.z_mm`은 **절대 물리 z**,
`preprocessing.z_mm`은 **training 폴더의 offset**입니다. 현재 설정은 각각 6과 0이며,
`z_reference_mm=6`을 기준으로 두 값의 일치를 검사합니다. Stage는 움직이지 않습니다.

기존 report의 shared affine/homography, exposure, ND transmission, saturation,
reference anchor 및 기록된 전처리 설정을 재사용합니다. 기본적으로 geometry를 다시
fit하지 않습니다. `allow_refit=true`는 report가 없을 때 명시적으로 새 calibration을
만드는 경우에만 사용하며, 기존 학습 좌표와 같다는 보장이 없어 기본값은 `false`입니다.
이때는 원래 전처리가 요구하는 `Forward_sim/z_*`와 calibration captures도 필요합니다.

Training에 dark/flat correction을 적용했다면 `preprocessing.options.dark_root` /
`flat_root`에 같은 calibration 자료를 지정해야 합니다. 기존 report는 HDR weight
threshold, 일부 registration QC 등의 모든 CLI 옵션을 기록하지 않습니다.
원래 실행에서 기본값을 바꿨다면 `options`에 해당 이름과 값을 넣으세요.
기록된 report 설정과 충돌하는 override는 거부하며, 기록이 없어 기본값을 사용한
항목과 현재 calibration 파일의 SHA256을 provenance에 남깁니다.

## 실행

기존 optical optimizer가 실행되는 Python 환경에 NumPy, PyTorch, OpenCV, SciPy,
Matplotlib과 실제 PC의 pypylon, hololith, SLM vendor DLL이 필요합니다.
학습된 intensity-response LUT와 동일한 카메라 gain/black-level/센서 설정을 사용하세요.
Camera 연결에서 auto exposure/gain을 끄고 BayerRG12와 수동 exposure를 적용합니다.

```powershell
# Update the lab checkout and create the local configuration on first use.
git pull --ff-only origin main
if (-not (Test-Path citl_config.local.json)) {
    Copy-Item citl_config.example.json citl_config.local.json
}

# Exercise the loop, saving, and plotting with a small simulated optical system.
python citl_closed_loop.py --dry-run --iters 5

# Validate calibration, checkpoint, phase, target, and gradient without hardware.
python citl_closed_loop.py --config citl_config.local.json --preflight

# Connect to the SLM, Arduino, and Basler for 30 phase updates.
python citl_closed_loop.py --config citl_config.local.json
```

`--iters 2` 또는 `--output-dir D:/CITL/runs`로 횟수/출력 폴더를 바꿀 수 있습니다.
`--iters 0`은 initial phase 측정만 수행합니다. 출력은 매번 새 timestamp 폴더입니다.
`--dry-run`은 실제 CSSNet이나 camera calibration의 정확도를 검증하는 기능은 아닙니다.
`--preflight`도 vendor SDK 설치·통신·광학 정렬을 실제로 시험하지는 않습니다.

## 각 iteration의 동작

1. 학습과 같은 fixed reference phase 업로드 → exposure bracket 촬영.
2. 현재 target phase 업로드 → 같은 exposure bracket 촬영.
3. Native Bayer DN → pixelwise inverse-response LUT → dark/flat → exposure/ND 및
   reference power correction → reference translation과 고정 affine을 합성한 warp → HDR.
4. 학습 checkpoint의 native crop, black-level, 고정 camera scale 적용.
5. 보정된 **실제 측정 intensity**로 loss 및 `dL/df` 계산.
6. 업로드한 양자화 phase에서 physical propagation + frozen CSSNet의
   `d(f_hat)/d(phase)`를 계산하고, `predicted.backward(dL_df)`로 결합.
7. Adam으로 phase만 update → wrap/선택적 변화량 제한 → 다음 SLM mask 생성.

SLM·Arduino·camera를 한 번만 열고 SLM memory slot 1을 반복 사용합니다.
각 exposure 전 acquisition queue를 다시 시작하고 stale frames를 버립니다.
DN averaging은 nonlinear camera LUT와 포화 판정을 바꾸므로 사용하지 않으며
`average_frames=1`이어야 합니다. 업로드·디스크 저장·최적화 중에는 laser를 끕니다.
종료/예외/Ctrl+C에서 laser off와 장비 close를 시도합니다.

Calibration/LUT/master arrays는 지속되는 별도 Python 프로세스에서 한 번 준비합니다.
이 프로세스는 PyTorch를 import하지 않아, Windows의 SciPy/MKL과 PyTorch OpenMP 충돌을
분리합니다. SLM/camera와 proxy는 주 프로세스에 유지되며 iteration마다 프로세스를
다시 띄우지 않습니다. Worker 오류는 주 프로세스에 전달되어 loop를 중단합니다.

SLM mask는 기존 optimizer의 transpose/flip 및 0..1023 ↔ 0..2π convention을
그대로 사용하고 hardware에서는 중앙 zero padding만 합니다. Camera LUT를 SLM에
적용하지 않습니다. 별도의 SLM nonlinear drive LUT/WFC는 추가하지 않으며, 현재
offline forward model과 같은 SLM 응답/보정 상태를 유지해야 합니다.

## Loss와 유효영역

`loss=shape`는 유효한 영역의 평균으로 target/measurement를 각각 나눈 smooth-L1로,
기존 offline shape loss와 같습니다. 상대 패턴을 맞추며 절대 밝기 보장은 없습니다.
`loss=mse`는 checkpoint camera units 그대로 비교합니다. Target 절대 밝기를
맞출 필요가 있으면 `target_peak`를 그 단위로 지정합니다. 매 iteration percentile
정규화는 하지 않습니다. 복잡한 training SSIM/FFT loss는 이 버전에서 지원하지 않습니다.

첫 측정의 valid mask를 고정 loss support로 저장합니다. 이후 같은 픽셀이 포화되거나
유효하지 않으면 해당 결과와 diagnostics를 저장하고 update 전에 중단합니다.
Loss 영역이 iteration마다 바뀌어 best 비교가 왜곡되는 것을 방지합니다.
`min_valid_fraction=0.9`는 crop 전체에서 필요한 최소 유효 비율입니다.
Confidence는 저장하지만 calibrated noise variance가 아니므로 loss weight로 사용하지 않습니다.

## 저장 결과

30 updates이면 초기 iteration 0부터 최종 iteration 30까지 **31개 측정**을 저장합니다.
Best와 final은 각각 실제 촬영해서 평가한 phase입니다.

```text
<output>/<timestamp>/
  run_config.json, provenance.json, hardware.json, status.json
  preprocessing_worker.log
  history.csv, loss_curve.png, comparison_latest.png
  target.npy, loss_support.npy
  phase_best_slm.npy, phase_final_slm.npy, phase_last_measured_slm.npy
  phase_best_rad_internal.npy, phase_final_rad_internal.npy
  camera_best.npy, camera_final.npy
  iterations/0000/ ... 0030/
    phase_slm.npy, phase_rad_internal.npy
    raw_reference/Exp300.raw, Exp1000.raw, Exp5000.raw
    raw_target/Exp300.raw, Exp1000.raw, Exp5000.raw
    aligned_hdr.npy, quality.npz, preprocessing.json, hardware.json
    camera_scaled.npy, proxy.npy, comparison.png
```

`aligned_hdr.npy`는 최종 alignment/HDR 보정 scientific intensity이며 invalid 픽셀은
NaN입니다. `quality.npz`의 `valid_mask`, `confidence`, `fallback_mask`가 동반됩니다.
`camera_scaled.npy`는 loss에 사용한 checkpoint 단위의 native crop입니다.
그림은 target/initial/current/proxy/best/error를 **고정 color range**로 보여줍니다.
`plot_every`를 늘리면 이미지·loss CSV는 계속 저장하면서 PNG 생성 빈도만 줄입니다.

`*_rad_internal.npy`는 quantization 전 연속 Adam 변수로, 내부 배열 방향입니다.
실제 재업로드에는 device/file 방향과 양자화가 적용된 `*_slm.npy`를 사용하세요.
중단된 run에는 `phase_final_slm.npy`를 만들지 않습니다. 자동 resume는 지원하지 않으며
`phase_last_measured_slm.npy` 또는 `phase_best_slm.npy`를 새 실행의 시작점으로 사용할 수 있습니다.

기본 raw 크기에서는 reference+target 3-exposure 한 측정에 약 107 MB가 필요합니다.
30 updates의 raw만 약 3.3 GB이고, 보정 결과/phase/PNG 저장 용량이 추가됩니다.

## 검증

```powershell
python -m pytest -q tests
python -m unittest discover -s tests -p test_citl_camera_preprocess.py -v
```

Camera numerical integration tests는 `CITL_PREPROCESS_SCRIPT` 환경변수 또는 저장소에
포함한 원본 전처리 경로를 사용합니다. 원본을 찾지 못하면 해당 integration tests만
skip됩니다. PyTorch가 이미 import된 test runner에서는 동일 프로세스의 numerical
tests만 skip하고, 별도 자식 프로세스에서 전체 전처리 검증을 반드시 실행합니다.
Worker tests는 PyTorch가 있는 주 프로세스와 실제 전처리 자식 프로세스의 연결도 검사합니다.
장비 mock 테스트와 numerical 테스트는 실제 camera/SLM 측정 성공을 대신하지 않습니다.
