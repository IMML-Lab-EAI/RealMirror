# RealMirror rollout annotation — 출력 포맷

`scripts/annotate_episode.py` / `scripts/annotate_batch.py` 가 episode 마다 아래 파일을 만든다.

```
outputs/realmirror_smolvla/<Task>/<run>/episode-XXXXXX/
├── episode_meta.json     # 카메라 intrinsics, 물체 목록(이름/카테고리/크기), mask id 표, 좌표계 규약
├── annotations.jsonl     # step 하나 = 한 줄(dict)
├── masks.h5              # id_map[step, camera_index, y, x] uint16 (--render 일 때)
└── review/step_XXXXXX.jpg  # 검토 이미지 (head | left_wrist | right_wrist), 2D bbox 만 표시
```

## annotations.jsonl 한 줄 (step)

```
step, success, instruction
cameras{name: position_world, quat_wxyz_world}          # 손목 카메라는 FK로 매 step 갱신
robot{root_position, root_quat_wxyz, joint_positions[50], state26}
grippers{left|right: mask_id, palm_position_world, palm_quat_wxyz, wrist_position_world,
         fingertips_world{thumb..little}, box3d_center_world, box3d_size_m, box3d_corners_world[8],
         closure_deg, closure_ratio, closed, finger_joint_angles_rad, views{cam: <view>}}
objects[ ]   # trajectory.hdf5 에 pose 가 기록된 task 물체
   name, prim_path, mask_id, category, role(target|destination|source_support|destination_basket|distractor), active,
   position_world, quat_wxyz, linear_velocity, size_m, box3d_center_world, box3d_corners_world[8],
   dist_to_left_palm_m, dist_to_right_palm_m, views{cam: <view>}
scene_objects[ ]   # 그 외 씬 물체(과일, 도마, 가구 등) 중 어느 카메라에든 1px 이상 보이는 것
   mask_id, name, category, background, prim_path, static(true),
   position_world(world AABB 중심), size_m, box3d_corners_world[8],
   dist_to_left_palm_m, dist_to_right_palm_m, views{cam: <view>}   # 보이는 카메라만
```

### `<view>` (카메라 하나에서의 2D 정보)

| 필드 | 의미 |
|---|---|
| `in_frustum` | 투영 기준으로 화면 안에 들어오는지 (가림 무시) |
| `visible`, `visible_px` | 렌더된 instance-ID 이미지에서 실제 보이는 픽셀 수 기준 |
| `bbox_xyxy_px` / `_norm` | 메시 정점 투영의 tight box (가림 무시) |
| `bbox_box3d_xyxy_px`, `corners_px` | 3D 박스 8꼭짓점 투영 |
| `bbox_visible_xyxy_px` / `_norm` | 실제 보이는 픽셀의 box |
| `bbox_visible_main_xyxy_px`, `visible_px_main`, `n_components` | 가장 큰 연결 성분만의 box (틈으로 보이는 조각 제외) |
| `mask_centroid_px`, `center_px`, `depth_m` | mask 중심, 3D 중심 투영점, 카메라로부터의 깊이 |
| `unoccluded_area_px_approx`, `occluded_fraction_approx` | 투영 정점 convex hull 면적 대비 보이는 픽셀 비율로 근사한 가림 |

좌표: 픽셀은 좌상단 원점 xyxy, 3D 는 Isaac world (m, z-up), quaternion 은 wxyz.

### masks.h5

`id_map[step, cam, y, x]` 값이 mask id. `episode_meta.json` 의 `scene_objects[].mask_id`, 스텝별 `annotations.jsonl`의 `objects[].mask_id`,
`mask_ids{left_hand:60001, right_hand:60002, robot_body:60003}` 로 해석한다. 0 은 아무것도 없음.

```python
import h5py, numpy as np
m = h5py.File('masks.h5')['id_map'][t, 0]        # head camera
lemon = m == 23                                  # scene_objects 에서 name=='lemon' 인 mask_id
```

## 카메라 모델 (검증됨)

* head: task JSON 의 position/orientation, horizontal_aperture, focal 50 mm (USD 기본값; Task1 캔 뚜껑 fit rms 2.1 px)
* wrist: 로봇 USD 의 `<side>_wrist_camera` Camera prim (focal 10 mm, aperture 15 mm) 이 `<side>_arm_link05` 에 fixed joint 로 붙어 있어 FK 로 매 step 계산
* Isaac `Camera.set_world_pose` 의 world axes (+x 전방, +y 왼쪽, +z 위) 규약, fy = fx

## 한계

* 씬의 정적 물체는 USD 에 적힌 초기 pose 로 고정된다고 가정한다 (로봇이 밀어 움직였다면 mask 가 어긋남).
* 손 3D 박스는 손가락 링크 메시의 palm-frame AABB 라 펼친 손에서는 넓다.
* `occluded_fraction_approx` 는 convex hull 근사값이다.
* **head 카메라 잔상**: 평가 스크립트가 (999,999,999)로 치워 둔 비활성 물체(Task1 병 2개, Task5 음식 2개)가 head 카메라 이미지에만 원래 배치 위치에 반투명하게 남아 보이는 경우가 있다. 손목 카메라에는 없고 기록된 pose도 999이므로 Isaac RTX 렌더 잔상으로 판단해 annotation 하지 않는다 (`objects[].active=false`).
