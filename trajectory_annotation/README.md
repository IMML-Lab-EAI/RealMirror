# RealMirror 궤적 주석 생성 파이프라인

RealMirror 평가 궤적의 매 스텝에 작업 물체·양손·주변 물체의 **3D 박스, 카메라별 2D 박스, instance mask, 가시성, 손과 물체 사이 거리**를 추가합니다. 원본 RGB와 action은 원본 HDF5에 유지하고, 주석을 별도로 저장합니다.

## 데이터셋을 만든 방법

1. [`evaluate/trajectory_recorder.py`](../evaluate/trajectory_recorder.py)가 기록한 물체 pose, 로봇 관절값, 카메라 영상, 작업 지시문과 성공 여부를 읽습니다.
2. RealMirror USD 자산에서 물체 크기·메시를 추출하고, YAML 설정으로 이름·카테고리·물체 분할 기준을 지정합니다. 정적 장면은 초기 위치로, 추적 물체는 매 스텝 기록된 pose로 복원합니다.
3. 로봇 URDF와 관절값으로 순기구학(FK)을 계산해 양손과 손목 카메라의 pose를 구하고, 3D 박스와 메시 정점을 세 카메라에 투영합니다.
4. `--render` 사용 시 **pyrender + GPU/EGL**로 instance ID를 렌더링해 실제 보이는 영역의 마스크·박스·픽셀 수를 계산합니다. 주석 생성 자체에는 Isaac Sim 실행이나 VLM/SAM 추론이 필요하지 않습니다.
5. 모든 스텝을 JSONL/HDF5로 저장하고, 에피소드의 처음·중간·마지막 이미지를 검토용으로 출력합니다.

기존 SmolVLA 데이터 생성 결과는 5개 작업, **1,500 에피소드 / 662,190 스텝 / 3개 카메라(256×256)**입니다. 이는 1,986,570개 카메라 프레임에 대응하며, 성공 1,118개와 실패 382개를 모두 포함합니다. 수치는 생성된 메타데이터 기준이며 데이터 파일 자체는 이 디렉터리에 포함하지 않습니다.

| 작업 | 에피소드 | 스텝 |
|---|---:|---:|
| Task1_Kitchen_Cleanup | 400 | 57,545 |
| Task2_Cup_to_Cup_Transfer | 200 | 54,711 |
| Task3_Assembly_Line_Sorting | 100 | 267,393 |
| Task4_Can_Stacking | 400 | 166,388 |
| Task5_Air_Fryer_Manipulation | 400 | 116,153 |

## 실행 방법

아래 명령은 모두 **이 디렉터리에서** 실행합니다. Python 3.9 이상을 사용하고, RealMirror/Isaac Sim과 분리된 가상환경 설치를 권장합니다. 마스크 생성에는 OpenGL/EGL을 지원하는 GPU와 드라이버가 필요합니다.

```bash
cd trajectory_annotation
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[assets,render,test]'
```

### 1. 자산과 평가 궤적 준비

[RealMirror 자산](https://huggingface.co/datasets/zte-terminators/realmirror-asset)을 다운로드하여 `data/realmirror-asset/` 아래에 `robot/`, `scenes/`가 오도록 배치합니다. 이미 다운로드했다면 다음처럼 연결합니다. `/absolute/path/...`는 실제 경로로 바꿉니다.

```bash
mkdir -p data
ln -s /absolute/path/to/realmirror-asset data/realmirror-asset
```

입력은 이 저장소의 평가 실행으로 얻은 **완료된 궤적**입니다. 물체 pose가 없는 LeRobot 시연 데이터는 직접 사용할 수 없습니다. `--runs`에는 `Task*` 디렉터리들을 바로 포함하는 경로를 지정합니다.

```text
<RUNS>/
└── Task1_Kitchen_Cleanup/
    └── smolvla_<timestamp>/
        └── trajectories/
            └── episode-000000/
                ├── trajectory.hdf5
                ├── metadata.json
                └── task_config.json
```

HDF5에서 `observations/{head,left_wrist,right_wrist}_camera_bgr`, `observations/state`, `states/objects/*`, `states/robot/*`, `objects/prim_paths`, `robot/dof_names`, `step_index`, `success`, `infos/json`을 읽습니다.

### 2. 물체 테이블·장면 캐시 생성 (최초 1회)

```bash
python scripts/build_object_tables.py
python scripts/build_scene_cache.py
```

`data/object_tables/`에 물체 크기·정점, `data/scene_cache/`에 장면 메시·ID 목록을 만듭니다. 두 명령 모두 `--task Task1_Kitchen_Cleanup`으로 작업 하나만 처리할 수 있습니다.

### 3. 에피소드 하나 확인

```bash
export PYOPENGL_PLATFORM=egl
EPISODE=/absolute/path/to/episode-000000
python scripts/annotate_episode.py \
  --episode "$EPISODE" --out outputs/sample \
  --render --review-count 3
```

빠른 확인은 `--max-steps 3`을 추가합니다. 이때 메타데이터의 `num_steps`와 마스크 배열 길이는 원본 길이를 유지하고 실제 주석은 앞의 3스텝만 생성되므로, 시험 출력과 전체 출력을 구분해야 합니다. `--render`를 생략하면 CPU에서 기하 주석만 생성하며 마스크·가림 기반 가시성은 제공하지 않습니다.

### 4. 전체 데이터셋 생성

```bash
RUNS=/absolute/path/to/evaluation_root
python scripts/annotate_batch.py \
  --runs "$RUNS" --out outputs/realmirror_smolvla \
  --render --review-count 3
```

- `--tasks Task1_Kitchen_Cleanup,Task2_Cup_to_Cup_Transfer`: 작업 선택
- `--only-success`: 성공 에피소드만 선택 (기본값은 성공·실패 모두)
- `--limit 2`: 처리할 에피소드 수 제한
- `--shard 0/2`, `--shard 1/2`: 서로 다른 GPU 작업에서 에피소드를 나누어 처리

완료 판정은 JSONL 줄 수와, 렌더링 시 마스크 파일 존재 여부를 기준으로 하므로 재실행 시 완료된 에피소드는 건너뜁니다. 손상된 마스크까지 검사하는 기능은 아니며, 오류 내역은 `batch_log_*.csv`에 기록합니다. 기존 전체 출력의 JSONL+마스크는 약 37.2 GB이며 원본 영상·자산 공간은 별도로 필요합니다.

## 출력 데이터셋 구조

```text
outputs/realmirror_smolvla/
└── <Task>/<run>/episode-XXXXXX/
    ├── episode_meta.json
    ├── annotations.jsonl
    ├── masks.h5
    └── review/step_XXXXXX.jpg
```

| 파일 | 내용 |
|---|---|
| `episode_meta.json` | 원본 `episode_dir`, 작업·최종 성공 여부·스텝 수, 카메라 내부 파라미터, 물체와 mask ID 목록 |
| `annotations.jsonl` | 한 줄이 한 스텝이며 `step`, `success`, `instruction`, `cameras`, `robot`, `grippers`, `objects`, `scene_objects` 포함 |
| `masks.h5` | `id_map[T, 3, 256, 256]`, `uint16`; `--render` 사용 시 생성 |
| `review/*.jpg` | 원본 세 카메라 영상을 가로로 연결하고 2D 박스를 표시한 검토 이미지 |

- `objects`: pose가 기록된 작업 물체. 이름·역할·활성 여부, pose·속도·3D 박스, 양손과 거리, 카메라별 `views`를 저장합니다.
- `grippers.left/right`: 손바닥·손목·손가락 말단 링크 위치, 손 전체 3D 박스, 관절각·닫힘 정도와 `views`를 저장합니다.
- `scene_objects`: 화면에 보이는 주변 정적 물체. `views`에는 보이는 카메라만 포함합니다.
- `views`: 투영 박스(`bbox_xyxy_px`), 실제 가시 박스(`bbox_visible_xyxy_px`), 가시 픽셀 수, 중심·깊이·근사 가림 비율입니다. 비활성/비가시 대상은 일부 필드가 없습니다.
- 카메라 순서는 `head_camera`, `left_wrist_camera`, `right_wrist_camera`입니다. 마스크 ID `0`은 대상 없음, `60001/60002/60003`은 왼손/오른손/로봇 몸체입니다. 일반 물체 ID는 에피소드 메타데이터의 `scene_objects`와 연결합니다.
- 3D 좌표는 미터·z-up, quaternion은 `wxyz`, 2D 박스는 좌상단 원점의 `[xmin, ymin, xmax, ymax]`입니다.

원본 RGB·action은 `episode_meta.json`의 `episode_dir` 아래 `trajectory.hdf5`에서 읽습니다. JSONL의 `step=t`는 원본 HDF5의 행 인덱스와 `id_map[t]`에 대응합니다. 손목 카메라의 실제 외부 파라미터는 스텝별 `cameras`를 사용합니다. 상세 필드는 [출력 포맷](docs/output_format.md)을 참고하세요.

## 코드와 검증

`realmirror_annot/`는 기하 계산·렌더링·주석·시각화 모듈, `configs/`는 물체 이름과 장면 분할 설정, `scripts/`는 캐시 생성·단일/배치 실행·기존 출력 보정 도구입니다.

```bash
python -m pytest -q
```

카메라 투영, 로봇 FK, instance ID 인코딩을 검증합니다. FK 테스트는 자산이 없으면 건너뜁니다. `redraw_reviews.py`, `fix_invisible_tracked.py`, `rename_in_outputs.py`는 기존 출력 파일을 수정하는 유지보수 도구이며 정상적인 신규 생성에는 필요하지 않습니다.

## 해석 시 주의점

- 정적 물체는 USD 초기 pose로 고정하고 일부 메시·투영 정점을 단순화/샘플링하므로, 렌더 마스크와 원본 RGB가 완전히 일치하지 않을 수 있습니다. 가림 비율도 근사값입니다.
- 물체 `role`은 첫 스텝의 작업 기준으로 고정됩니다. Task3처럼 중간에 조작 대상이 바뀌는 경우 현재 대상 라벨로 사용하려면 별도 갱신이 필요합니다.
- 손의 `closed`는 검지 관절각 기반 판정이며 접촉이나 파지 성공의 정답이 아닙니다.
- 비활성 물체는 멀리 치워진 pose를 기준으로 제외합니다. 원본 head 영상의 렌더 잔상까지 주석 처리하지 않습니다.
