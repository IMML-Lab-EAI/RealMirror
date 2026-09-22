# RealMirror 로봇 작업 데이터에 주석 붙이기

이 코드는 **로봇이 작업하는 동안 무엇이 어디에 있었는지**를 데이터로 정리합니다. 예를 들어 “과자통을 바구니에 넣기” 영상에 과자통과 양손의 위치, 화면에서 보이는 영역, 손과 과자통 사이 거리 등을 붙입니다. 이렇게 영상이나 기록에 설명 정보를 붙이는 작업을 **주석(annotation)**이라고 합니다.

입력은 RealMirror 시뮬레이터에 저장된 로봇 작업 기록입니다. 머리·왼쪽 손목·오른쪽 손목 카메라를 모두 처리하며, 원본 영상과 로봇 행동 명령은 그대로 두고 주석 파일을 따로 만듭니다.

- **에피소드(episode)**: 로봇이 작업을 한 번 시도한 기록
- **스텝(step)**: 그 기록의 한 시점. 스텝 하나에 세 카메라 영상과 로봇·물체 상태가 들어 있습니다.
- **2D 박스**: 영상에서 물체를 둘러싼 사각형
- **3D 박스**: 실제 3차원 공간에서 물체를 둘러싼 상자
- **마스크(mask)**: 영상의 각 픽셀이 어느 물체에 속하는지 표시한 지도

## 우리가 붙인 주석은 무엇인가요?

### 주석을 붙인 대상

| 대상 | 포함하는 것 | 저장 위치 |
|---|---|---|
| 작업 관련 물체 | 로봇이 옮기는 물체, 목적지 바구니·컵, 받침대 등 | `objects` |
| 로봇의 양손 | 왼손·오른손, 손바닥·손목·각 손가락 끝부분 | `grippers.left`, `grippers.right` |
| 주변 환경 | 카메라에 보이는 과일·식기·가구·가전·벽 등 | `scene_objects` |

작업별로 위치를 추적하는 물체는 다음과 같습니다. 한 에피소드에서 사용하지 않는 물체는 `active=false`로 표시될 수 있습니다.

| 작업 | 주요 주석 대상 |
|---|---|
| 주방 정리 | 과자통, 녹차병, 레몬차병, 노란 바구니 |
| 컵 사이 옮기기 | 딸기, 왼쪽 컵, 오른쪽 컵 |
| 조립 라인 분류 | 기름병·콜라병·Sprite 병, 병 받침대, 좌우 상자, 컨베이어 롤러 |
| 캔 쌓기 | 왼쪽·오른쪽의 파란 캔과 빨간 캔 |
| 에어프라이어 조작 | 치킨롤·닭다리·당근, 접시, 에어프라이어 본체와 바구니 |

### 대상마다 저장한 정보

| 주석 정보 | 쉽게 설명하면 | 대표 필드 |
|---|---|---|
| 이름과 종류 | 어떤 물체인가? 음식·용기·가구 중 무엇인가? | `name`, `category` |
| 작업에서의 역할 | 옮길 대상인가, 놓을 곳인가, 받침대인가? | `objects[].role` |
| 위치·방향·크기 | 3D 공간의 어디에 있고 어느 방향을 향하는가? | `position_world`, `quat_wxyz`, `size_m` |
| 움직임 | 기록된 작업 물체가 어느 방향으로 얼마나 빠르게 움직이는가? | `objects[].linear_velocity` |
| 3D 박스 | 물체나 손이 차지하는 공간은 어디인가? | `box3d_corners_world` |
| 카메라별 2D 박스 | 각 영상에서 물체나 손이 어디에 있는가? | `views.*.bbox_xyxy_px` |
| 실제 보이는 영역 | 다른 물체에 가려지지 않고 보이는 부분은 어디인가? | `bbox_visible_xyxy_px`, `masks.h5` |
| 보이는 정도 | 화면에서 보이는가, 몇 픽셀인가, 얼마나 가려졌는가? | `visible`, `visible_px`, `occluded_fraction_approx` |
| 손과의 거리 | 물체의 박스 중심에서 왼쪽·오른쪽 손바닥까지 얼마나 떨어져 있는가? | `dist_to_left_palm_m`, `dist_to_right_palm_m` |
| 손의 자세 | 손바닥·손목·손가락 끝부분은 어디에 있고, 손이 얼마나 닫혀 있는가? | `palm_position_world`, `fingertips_world`, `closure_ratio`, `closed` |

위치·방향 등은 대상에 따라 필드가 다릅니다. 주변 정적 물체는 박스 중심과 크기를 저장하고, 양손은 손바닥의 위치·방향을 저장합니다. 마스크와 실제 보이는 영역 정보는 `--render` 옵션을 사용했을 때 생성됩니다.

예를 들어 과자통을 바구니에 넣는 에피소드에서는 과자통에 `target`(옮길 대상), 바구니에 `destination`(놓을 곳)을 붙입니다. 매 스텝마다 두 손과 과자통의 박스, 각 카메라에서 보이는 부분, 과자통과 양손 사이 거리도 함께 기록합니다.

작업 지시문(`instruction`), 성공 여부(`success`), 로봇 관절값(`robot`), 카메라 위치·방향(`cameras`)도 원본 기록에서 가져오거나 계산하여 함께 저장합니다. 다만 **손이 닫혔다는 정보가 물체를 잡는 데 성공했다는 뜻은 아닙니다.**

## 데이터셋은 어떻게 만들었나요?

1. **기존 작업 기록을 읽습니다.** [`trajectory_recorder.py`](../evaluate/trajectory_recorder.py)가 저장한 영상, 물체의 위치·방향, 로봇 관절값을 가져옵니다.
2. **물체의 모양과 크기를 준비합니다.** 시뮬레이터의 3D 모델 파일(USD)에서 모양을 읽고, 설정 파일(YAML)에 적힌 이름과 종류를 연결합니다.
3. **각 시점의 장면을 다시 구성합니다.** 물체는 기록된 위치로 옮깁니다. 손과 손목 카메라의 위치는 로봇 구조 파일(URDF)과 관절각으로 계산합니다. 이 계산을 순기구학(FK)이라고 합니다.
4. **카메라에서 어떻게 보이는지 계산합니다.** 3D 위치를 영상 좌표로 바꿔 박스를 만듭니다. `--render`를 사용하면 장면을 다시 그려 물체끼리 가리는 부분까지 반영한 마스크를 만듭니다. 여기에 `pyrender`와 GPU/EGL을 사용합니다.
5. **모든 스텝의 결과를 저장합니다.** 검토용으로 처음·중간·마지막 이미지도 만듭니다. 검토 이미지가 3장이어도 주석은 모든 스텝에 생성됩니다.

주석의 바탕은 시뮬레이터에 기록된 위치와 3D 모델입니다. 주석을 만들 때 Isaac Sim을 다시 실행하거나 VLM/SAM으로 물체를 추측할 필요는 없습니다. 복원한 장면과 원본 영상 사이에는 차이가 있을 수 있으며, 아래의 주의점에 설명했습니다.

## 지금까지 만든 데이터 규모

기존 SmolVLA 데이터 생성 결과는 5개 작업, **1,500 에피소드 / 662,190 스텝 / 3개 카메라(256×256)**입니다. 이는 1,986,570개 카메라 프레임에 대응하며, 성공 1,118개와 실패 382개를 모두 포함합니다. 수치는 생성된 메타데이터 기준이며 데이터 파일 자체는 이 디렉터리에 포함하지 않습니다.

| 작업 폴더 | 하는 일 | 에피소드 | 스텝 |
|---|---|---:|---:|
| Task1_Kitchen_Cleanup | 과자통이나 음료병을 바구니에 넣기 | 400 | 57,545 |
| Task2_Cup_to_Cup_Transfer | 오른쪽 컵의 딸기를 왼쪽 컵으로 옮기기 | 200 | 54,711 |
| Task3_Assembly_Line_Sorting | 기름병·콜라를 지정 위치로 옮기고 Sprite 유지하기 | 100 | 267,393 |
| Task4_Can_Stacking | 같은 색 캔을 옮겨 쌓기 | 400 | 166,388 |
| Task5_Air_Fryer_Manipulation | 음식을 에어프라이어에 넣기 | 400 | 116,153 |

## 실행 방법

Python 3.9 이상이 필요합니다. 마스크를 만들려면 OpenGL/EGL을 지원하는 GPU와 드라이버도 필요합니다. 저장소 최상위에서 아래 명령으로 시작한 뒤, 나머지 명령은 모두 **`trajectory_annotation/` 안에서** 실행합니다. 가상환경을 사용하면 RealMirror/Isaac Sim의 기존 설치와 분리할 수 있습니다.

```bash
cd trajectory_annotation
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[assets,render,test]'
```

### 1. 3D 모델과 원본 작업 기록 준비

[RealMirror 자산](https://huggingface.co/datasets/zte-terminators/realmirror-asset)을 다운로드하여 `data/realmirror-asset/` 아래에 `robot/`, `scenes/`가 오도록 배치합니다. 이미 다운로드했다면 다음처럼 연결합니다. `/absolute/path/...`는 실제 경로로 바꿉니다.

```bash
mkdir -p data
ln -s /absolute/path/to/realmirror-asset data/realmirror-asset
```

입력은 이 저장소에서 로봇 평가를 끝낸 뒤 저장된 **작업 기록**입니다. 물체의 위치·방향이 없는 LeRobot 시연 데이터는 직접 사용할 수 없습니다. `--runs`에는 아래처럼 `Task*` 폴더들을 바로 포함하는 경로를 지정합니다.

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

`trajectory.hdf5`는 영상과 로봇·물체 상태를 담은 파일입니다. `metadata.json`은 작업의 요약 정보, `task_config.json`은 카메라와 작업 설정을 담고 있습니다.

### 2. 물체의 크기와 장면 모양을 미리 저장하기 (최초 1회)

```bash
python scripts/build_object_tables.py
python scripts/build_scene_cache.py
```

첫 번째 명령은 `data/object_tables/`에 물체의 크기와 모양을 나타내는 점들을 저장합니다. 두 번째는 `data/scene_cache/`에 장면의 3D 모양과 물체 번호를 저장합니다. 이후 에피소드를 처리할 때 이 준비된 파일을 재사용합니다. 두 명령 모두 `--task Task1_Kitchen_Cleanup`을 붙이면 작업 하나만 처리합니다.

### 3. 작업 기록 하나에 주석 붙여 보기

```bash
export PYOPENGL_PLATFORM=egl
EPISODE=/absolute/path/to/episode-000000
python scripts/annotate_episode.py \
  --episode "$EPISODE" --out outputs/sample \
  --render --review-count 3
```

결과는 `outputs/sample/`에 생깁니다. `review/`의 이미지를 열어 박스 위치를 확인할 수 있습니다.

빠르게 시험하려면 `--max-steps 3`을 붙여 앞의 3스텝만 처리합니다. 이때 `num_steps`와 마스크 배열 길이는 원본 전체 길이로 남으므로, 시험 결과는 별도 출력 폴더에 저장하세요. GPU가 없으면 `--render`를 빼고 위치·박스 계산만 실행할 수 있습니다. 이 경우 실제 가려짐을 반영한 마스크는 생성하지 않습니다.

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

다시 실행하면 완료된 에피소드는 건너뜁니다. 완료 여부는 주석 줄 수와 마스크 파일의 존재로 판단하며, 마스크 파일 손상까지 검사하지는 않습니다. 처리 결과와 오류는 `batch_log_*.csv`에서 확인할 수 있습니다. 기존 전체 주석과 마스크의 크기는 약 37.2 GB이며, 원본 영상과 3D 모델은 별도 공간이 필요합니다.

## 결과 파일은 어떻게 구성되나요?

폴더는 **작업 → 실행 기록(run) → 에피소드** 순서입니다. 에피소드 하나에는 다음 파일이 들어 있습니다.

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
| `episode_meta.json` | 에피소드 설명서: 원본 경로, 작업 이름, 최종 성공 여부, 스텝 수, 카메라 설정, 물체 번호 목록 |
| `annotations.jsonl` | 한 줄에 한 스텝의 주석. 한 줄 안에 세 카메라의 정보가 모두 들어 있음 |
| `masks.h5` | 각 픽셀에 물체 번호를 저장한 마스크. `--render` 사용 시 생성 |
| `review/*.jpg` | 세 카메라 영상을 가로로 이어 붙이고 박스를 그린 검토 이미지 |

`annotations.jsonl`의 한 줄은 다음처럼 구성됩니다. 아래는 필드 구조를 설명한 것으로, 실제 값은 스텝마다 달라집니다.

```text
한 스텝의 주석
├── step                    # 몇 번째 스텝인지 (0부터 시작)
├── instruction / success   # 작업 지시문 / 이 스텝의 성공 여부
├── cameras                 # 세 카메라의 위치와 방향
├── robot                   # 로봇 위치와 관절값
├── grippers.left / right   # 양손의 주석
├── objects                 # 작업 물체의 주석
└── scene_objects           # 보이는 주변 물체의 주석
```

물체와 양손의 `views` 안에는 카메라별 박스와 보이는 정도가 들어 있습니다. 주변 물체는 보이는 카메라만 기록합니다. 사용하지 않거나 보이지 않는 물체에는 일부 필드가 없을 수 있습니다.

마스크는 `id_map[T, 3, 256, 256]` 형태의 `uint16` 배열입니다. `T`는 스텝 수, `3`은 머리·왼쪽 손목·오른쪽 손목 카메라 순서입니다. 픽셀값 `0`은 대상 없음, `60001/60002/60003`은 왼손/오른손/나머지 로봇 몸체를 뜻합니다. 일반 물체 번호는 `episode_meta.json`의 `scene_objects`에서 찾습니다. 예를 들어 어떤 물체의 `mask_id`가 23이면 `id_map[t, 0] == 23`이 스텝 `t`의 머리 영상에서 그 물체의 영역입니다.

**원본 영상과 연결하기:** `episode_meta.json`의 `episode_dir`가 원본 폴더를 가리킵니다. 그 안의 `trajectory.hdf5`에서 영상과 행동 명령을 읽고, 주석의 `step=t`를 원본의 `t`번째 행 및 마스크의 `id_map[t]`와 연결하면 됩니다. 손목 카메라는 움직이므로 위치·방향은 각 스텝의 `cameras`를 사용합니다.

좌표와 필드의 정확한 정의는 [출력 포맷](docs/output_format.md)에 있습니다. 기본적으로 3D 길이 단위는 미터이고 z축이 위쪽입니다. 2D 박스는 영상 왼쪽 위를 원점으로 한 `[왼쪽 x, 위쪽 y, 오른쪽 x, 아래쪽 y]`, 회전은 `wxyz` 순서의 quaternion으로 저장합니다.

## 코드와 검증

`realmirror_annot/`는 기하 계산·렌더링·주석·시각화 모듈, `configs/`는 물체 이름과 장면 분할 설정, `scripts/`는 캐시 생성·단일/배치 실행·기존 출력 보정 도구입니다.

```bash
python -m pytest -q
```

카메라 좌표 계산, 관절값으로 계산한 손 위치, 마스크 물체 번호 변환을 검증합니다. 로봇 위치 테스트는 3D 모델이 없으면 건너뜁니다. `redraw_reviews.py`, `fix_invisible_tracked.py`, `rename_in_outputs.py`는 기존 결과를 수정하는 유지보수 도구이며 처음 생성할 때는 실행하지 않아도 됩니다.

## 해석 시 주의점

- **원본 영상과 약간 다를 수 있습니다.** 주변 물체는 처음 위치에 고정되어 있다고 가정하고, 계산량을 줄이기 위해 일부 3D 모양을 단순화합니다. 실제로 물체가 밀려 움직였다면 마스크가 어긋날 수 있습니다. 가려진 비율도 근사값입니다.
- **작업 역할은 시작 시점 기준입니다.** `role`은 첫 스텝에서 정한 값을 유지합니다. Task3처럼 도중에 조작 대상이 바뀌면 현재 대상과 다를 수 있습니다.
- **손 닫힘과 잡기 성공은 다릅니다.** `closed`는 검지 관절각으로 판단합니다. 손과 물체의 거리 역시 실제 접촉 여부를 직접 알려주지는 않습니다. `fingertips_world`는 손가락 끝부분 링크의 기준점이며 접촉점을 뜻하지 않습니다.
- **사용하지 않는 물체는 제외합니다.** 시뮬레이터가 멀리 치워 둔 물체는 비활성으로 처리합니다. 원본 머리 카메라 영상에 잔상이 남아 있더라도 주석을 붙이지 않습니다.
