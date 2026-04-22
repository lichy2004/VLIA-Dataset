import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(".")

from src.utils.data_format import get_vlia_data_template, image_encoding, save_vlia_hdf5
from src.utils.joints import HAND_KEYPOINT_NAMES
from src.utils.transform import create_rotation_transform


HOLOASSIST_26_TO_MANO21 = {
    "wrist": 1,
    "thumb1": 2,
    "thumb2": 3,
    "thumb3": 4,
    "thumb4": 5,
    "index1": 6,
    "index2": 7,
    "index3": 8,
    "index4": 10,
    "middle1": 11,
    "middle2": 12,
    "middle3": 13,
    "middle4": 15,
    "ring1": 16,
    "ring2": 17,
    "ring3": 18,
    "ring4": 20,
    "pinky1": 21,
    "pinky2": 22,
    "pinky3": 23,
    "pinky4": 25,
}


def _parse_float_line(line):
    line = line.strip().split("\t")
    return [float(x) for x in line]


def _safe_load_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [line.rstrip("\n") for line in f]


def _pad_or_trim(arr, frame_count, fill_value=0.0):
    if arr.shape[0] == frame_count:
        return arr
    if arr.shape[0] > frame_count:
        return arr[:frame_count]

    pad_shape = (frame_count - arr.shape[0],) + arr.shape[1:]
    pad = np.full(pad_shape, fill_value, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def process_text(data):
    data["text"] = ""


def process_images(data, mp4_path):
    cap = cv2.VideoCapture(mp4_path)

    encode_data = []
    max_len = 0
    num_frames = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        jpeg_data = image_encoding(frame_bgr)
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
        num_frames += 1

    cap.release()

    if num_frames == 0:
        data["images"] = []
        return 0, 0

    padded_data = [d.ljust(max_len, b"\0") for d in encode_data]
    data["images"] = padded_data
    return max_len, num_frames


def process_metadata(data, frame_count, session_path):
    data["metadata"]["frames"] = frame_count
    data["metadata"]["data_path"] = session_path


def process_camera(data, export_dir, frame_count):
    intrinsic_path = os.path.join(export_dir, "Video", "Intrinsics.txt")
    pose_path = os.path.join(export_dir, "Video", "Pose_sync.txt")
    timing_path = os.path.join(export_dir, "Video", "VideoMp4Timing.txt")

    intrinsic_lines = _safe_load_lines(intrinsic_path)
    if intrinsic_lines:
        vals = _parse_float_line(intrinsic_lines[0])
        data["camera"]["intrinsic"] = np.array(vals[:9], dtype=np.float32).reshape(3, 3)
    else:
        data["camera"]["intrinsic"] = np.eye(3, dtype=np.float32)

    pose_lines = _safe_load_lines(pose_path)
    pose_tfs = []
    pose_ts = []
    for line in pose_lines:
        vals = _parse_float_line(line)
        if len(vals) < 18:
            continue
        pose_ts.append(int(vals[1]))
        tf = np.array(vals[2:18], dtype=np.float32).reshape(4, 4)
        pose_tfs.append(tf)

    if len(pose_tfs) == 0:
        camera_tfs = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
        camera_ts = np.arange(frame_count, dtype=np.int64)
        data["camera"]["extrinsic"] = camera_tfs
        return camera_ts

    pose_tfs = np.asarray(pose_tfs, dtype=np.float32)
    pose_ts = np.asarray(pose_ts, dtype=np.int64)

    timing_lines = _safe_load_lines(timing_path)
    camera_ts = []
    for line in timing_lines[:frame_count]:
        line = line.strip()
        if line:
            camera_ts.append(int(float(line)))
    camera_ts = np.asarray(camera_ts, dtype=np.int64)

    if camera_ts.shape[0] == 0:
        camera_ts = pose_ts.copy()

    if camera_ts.shape[0] < frame_count:
        if camera_ts.shape[0] == 0:
            pad_vals = np.arange(frame_count, dtype=np.int64)
            camera_ts = pad_vals
        else:
            pad = np.repeat(camera_ts[-1], frame_count - camera_ts.shape[0])
            camera_ts = np.concatenate([camera_ts, pad], axis=0)
    elif camera_ts.shape[0] > frame_count:
        camera_ts = camera_ts[:frame_count]

    pose_idx = _nearest_indices(pose_ts, camera_ts)
    camera_tfs = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
    valid_frames = np.where(pose_idx >= 0)[0]
    if valid_frames.shape[0] > 0:
        camera_tfs[valid_frames] = pose_tfs[pose_idx[valid_frames]]

    data["camera"]["extrinsic"] = camera_tfs
    return camera_ts


def _load_hand_joint_tfs(hand_path):
    lines = _safe_load_lines(hand_path)

    hand_ts = []
    hand_joints = []
    hand_valid = []
    eye4 = np.eye(4, dtype=np.float32)

    for line in lines:
        vals = _parse_float_line(line)
        if len(vals) < 2 + 469:
            continue

        ts = int(vals[1])
        hand_vec = np.array(vals[2:], dtype=np.float32)
        if hand_vec.size != 469:
            continue

        raw = hand_vec[1:-52]
        if raw.size != 26 * 16:
            continue

        joint26 = raw.reshape(26, 4, 4)
        valid = 1

        if np.isnan(joint26).any() or np.isinf(joint26).any():
            valid = 0
            joint26 = np.tile(eye4[None, :, :], (26, 1, 1))
        elif np.array_equal(joint26[1], joint26[5]):
            valid = 0

        hand_ts.append(ts)
        hand_joints.append(joint26.astype(np.float32))
        hand_valid.append(valid)

    if len(hand_joints) == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 26, 4, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    return (
        np.asarray(hand_ts, dtype=np.int64),
        np.asarray(hand_joints, dtype=np.float32),
        np.asarray(hand_valid, dtype=np.int32),
    )


def _nearest_indices(src_ts, dst_ts):
    if src_ts.shape[0] == 0:
        return np.full((dst_ts.shape[0],), -1, dtype=np.int64)

    src_ts = src_ts.astype(np.int64)
    dst_ts = dst_ts.astype(np.int64)
    insert_idx = np.searchsorted(src_ts, dst_ts)
    insert_idx = np.clip(insert_idx, 0, src_ts.shape[0] - 1)

    prev_idx = np.clip(insert_idx - 1, 0, src_ts.shape[0] - 1)
    next_idx = insert_idx

    prev_dist = np.abs(src_ts[prev_idx] - dst_ts)
    next_dist = np.abs(src_ts[next_idx] - dst_ts)
    use_prev = prev_dist <= next_dist
    return np.where(use_prev, prev_idx, next_idx)


def process_hand_keypoint(data, export_dir, camera_ts):
    frame_count = data["metadata"]["frames"]
    left_path = os.path.join(export_dir, "Hands", "Left_sync.txt")
    right_path = os.path.join(export_dir, "Hands", "Right_sync.txt")

    left_ts, left_joint26_all, left_valid_all = _load_hand_joint_tfs(left_path)
    right_ts, right_joint26_all, right_valid_all = _load_hand_joint_tfs(right_path)

    left_idx = _nearest_indices(left_ts, camera_ts)
    right_idx = _nearest_indices(right_ts, camera_ts)

    left_valid = np.zeros((frame_count,), dtype=np.int32)
    right_valid = np.zeros((frame_count,), dtype=np.int32)

    left_joint26 = np.tile(np.eye(4, dtype=np.float32)[None, None, :, :], (frame_count, 26, 1, 1))
    right_joint26 = np.tile(np.eye(4, dtype=np.float32)[None, None, :, :], (frame_count, 26, 1, 1))

    valid_left_frames = np.where(left_idx >= 0)[0]
    if valid_left_frames.shape[0] > 0:
        src_idx = left_idx[valid_left_frames]
        left_joint26[valid_left_frames] = left_joint26_all[src_idx]
        left_valid[valid_left_frames] = left_valid_all[src_idx]

    valid_right_frames = np.where(right_idx >= 0)[0]
    if valid_right_frames.shape[0] > 0:
        src_idx = right_idx[valid_right_frames]
        right_joint26[valid_right_frames] = right_joint26_all[src_idx]
        right_valid[valid_right_frames] = right_valid_all[src_idx]

    data["mask"]["hand"] = np.stack([left_valid, right_valid], axis=1).astype(np.int32)

    joint_tfs_world = {}
    for hand_name in HAND_KEYPOINT_NAMES:
        idx = HOLOASSIST_26_TO_MANO21[hand_name]
        joint_tfs_world[f"left_{hand_name}"] = left_joint26[:, idx]
        joint_tfs_world[f"right_{hand_name}"] = right_joint26[:, idx]
    joint_tfs_world["camera"] = data["camera"]["extrinsic"]

    # global coordinate rotation
    camera_tfs_first = joint_tfs_world["camera"][0:1]
    camera_tfs_first = (
        camera_tfs_first
        @ create_rotation_transform("x", -90)
        @ create_rotation_transform("z", 90)
    )

    # first frame align
    joint_tfs_camera = {}
    camera_tfs_first_inv = np.linalg.inv(camera_tfs_first)
    for joint_name, joint_tf in joint_tfs_world.items():
        joint_tfs_camera[joint_name] = camera_tfs_first_inv @ joint_tf

    # local coordinate rotation
    joint_tfs_camera["camera"] = (
        joint_tfs_camera["camera"]
        @ create_rotation_transform("x", -90)
        @ create_rotation_transform("z", 90)
    )
    joint_tfs_camera["left_wrist"] = (
        joint_tfs_camera["left_wrist"]
        @ create_rotation_transform("x", -90)
    )
    joint_tfs_camera["right_wrist"] = (
        joint_tfs_camera["right_wrist"]
        @ create_rotation_transform("y", 180)
        @ create_rotation_transform("x", 90)
    )

    # save hand keypoint
    left_hand_tfs = {}
    right_hand_tfs = {}
    for hand_name in HAND_KEYPOINT_NAMES:
        left_hand_tfs[hand_name] = joint_tfs_camera[f"left_{hand_name}"]
        right_hand_tfs[hand_name] = joint_tfs_camera[f"right_{hand_name}"]

    data["hand"]["keypoint"]["left"] = left_hand_tfs
    data["hand"]["keypoint"]["right"] = right_hand_tfs
    data["camera"]["extrinsic"] = joint_tfs_camera["camera"]


def process_gaze(data, export_dir, camera_ts):
    frame_count = data["metadata"]["frames"]
    intrinsic = data["camera"]["intrinsic"]
    camera_tfs = data["camera"]["extrinsic"]

    gaze = np.zeros((frame_count, 2), dtype=np.float32)
    gaze_mask = np.zeros((frame_count, 1), dtype=np.int32)

    eyes_path = os.path.join(export_dir, "Eyes", "Eyes_sync.txt")
    lines = _safe_load_lines(eyes_path)

    gaze_ts = []
    gaze_points = []
    gaze_valid = []

    for line in lines:
        vals = _parse_float_line(line)
        if len(vals) < 9:
            continue

        eye_origin_world = np.array(vals[2:5], dtype=np.float32)
        eye_dir_world = np.array(vals[5:8], dtype=np.float32)
        valid = int(vals[8])

        if np.isnan(eye_origin_world).any() or np.isnan(eye_dir_world).any():
            valid = 0

        gaze_ts.append(int(vals[1]))
        gaze_points.append(np.concatenate([eye_origin_world, eye_dir_world]).astype(np.float32))
        gaze_valid.append(valid)

    if len(gaze_ts) == 0:
        data["gaze"] = gaze
        data["mask"]["gaze"] = gaze_mask
        return

    gaze_ts = np.asarray(gaze_ts, dtype=np.int64)
    gaze_points = np.asarray(gaze_points, dtype=np.float32)
    gaze_valid = np.asarray(gaze_valid, dtype=np.int32)

    gaze_idx = _nearest_indices(gaze_ts, camera_ts)
    axis_transform = np.linalg.inv(
        np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    )

    height = 504.0
    width = 896.0

    for i in range(frame_count):
        idx = gaze_idx[i]
        if idx < 0:
            continue
        if gaze_valid[idx] == 0:
            continue

        eye_origin_world = gaze_points[idx, :3]
        eye_dir_world = gaze_points[idx, 3:]

        norm = np.linalg.norm(eye_dir_world)
        if norm < 1e-6:
            continue

        eye_dir_world = eye_dir_world / norm
        gaze_world = eye_origin_world + eye_dir_world * 0.5

        world_h = np.array([gaze_world[0], gaze_world[1], gaze_world[2], 1.0], dtype=np.float32)
        cam_inv = np.linalg.inv(camera_tfs[i])
        gaze_cam_h = axis_transform @ cam_inv @ world_h

        z = float(gaze_cam_h[2])
        if z <= 1e-6:
            continue

        x = float(gaze_cam_h[0] / z)
        y = float(gaze_cam_h[1] / z)
        u = intrinsic[0, 0] * x + intrinsic[0, 2]
        v = intrinsic[1, 1] * y + intrinsic[1, 2]

        u = float(np.clip(u, 0.0, width - 1.0))
        v = float(np.clip(v, 0.0, height - 1.0))

        gaze[i] = np.array([v, u], dtype=np.float32)
        gaze_mask[i, 0] = 1

    data["gaze"] = gaze
    data["mask"]["gaze"] = gaze_mask


def process_episode(session_dir, output_path):
    export_dir = os.path.join(session_dir, "Export_py")
    mp4_path = os.path.join(export_dir, "Video_compress.mp4")
    data = get_vlia_data_template()

    len_high, frame_count = process_images(data, mp4_path)
    process_metadata(data, frame_count, session_dir)
    process_text(data)
    camera_ts = process_camera(data, export_dir, frame_count)
    process_hand_keypoint(data, export_dir, camera_ts)
    process_gaze(data, export_dir, camera_ts)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_vlia_hdf5(output_path, data, len_high=len_high if len_high > 0 else None)


def collect_files(src_dir):
    session_dirs = []
    if not os.path.exists(src_dir):
        print(f"Path does not exist: {src_dir}")
        return session_dirs

    for name in os.listdir(src_dir):
        session_dir = os.path.join(src_dir, name)
        if not os.path.isdir(session_dir):
            continue

        export_dir = os.path.join(session_dir, "Export_py")
        if not os.path.isdir(export_dir):
            continue

        required = [
            os.path.join(export_dir, "Video", "Pose_sync.txt"),
            os.path.join(export_dir, "Hands", "Left_sync.txt"),
            os.path.join(export_dir, "Hands", "Right_sync.txt"),
            os.path.join(export_dir, "Video_compress.mp4"),
        ]
        if all(os.path.exists(p) for p in required):
            session_dirs.append(session_dir)

    session_dirs.sort()
    print(f"Found {len(session_dirs)} HoloAssist sessions to process.")
    return session_dirs


def main(raw_dir, out_dir):
    session_dirs = collect_files(raw_dir)

    for index, session_dir in tqdm(
        enumerate(session_dirs), desc="Processing HoloAssist Dataset"
    ):
        dst_path = os.path.join(out_dir, f"{index:08d}.hdf5")
        process_episode(session_dir, dst_path)


if __name__ == "__main__":
    src_dir = "data/HoloAssist/raw"
    dst_dir = "data/HoloAssist/processed"
    main(src_dir, dst_dir)
