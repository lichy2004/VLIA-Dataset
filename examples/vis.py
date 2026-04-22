import sys
sys.path.append('.')

import h5py
import numpy as np

from src.utils.data_format import image_decoding
from src.utils.joints import HAND_KEYPOINT_NAMES
from src.utils.visualize import vis_3d_keypoint, vis_2d_keypoint


def load_vlia_hdf5(file_path, max_frames=300):
    """Load VLIA hdf5 and return global camera/hand/mask/gaze data.

    Args:
        file_path: HDF5 file path.
        max_frames: Maximum number of frames to load for visualization.
                    Set to None to load all frames.
    """
    with h5py.File(file_path, 'r') as f:
        total_frames = f['images'].shape[0]
        num_frames = total_frames if max_frames is None else min(total_frames, max_frames)

        encoded_images = f['images'][:num_frames]
        imgs = np.stack([image_decoding(b, to_rgb=True) for b in encoded_images], axis=0)

        cam_int = np.asarray(f['camera']['intrinsic'][:num_frames], dtype=np.float32)
        cam_ext = np.asarray(f['camera']['extrinsic'][:num_frames], dtype=np.float32)

        keypoint_grp = f['hand']['keypoint']
        hand = {
            side: {
                name: np.asarray(keypoint_grp[side][name][:num_frames], dtype=np.float32)
                for name in HAND_KEYPOINT_NAMES
            }
            for side in ('left', 'right')
        }

        # Optional hand visibility mask and gaze.
        # mask format: {'left': [T], 'right': [T]}
        mask = {'left': np.ones(num_frames, dtype=bool), 'right': np.ones(num_frames, dtype=bool)}
        if 'mask' in f:
            mask_grp = f['mask']
            for side in ('left', 'right'):
                if side in mask_grp:
                    mask[side] = np.asarray(mask_grp[side][:num_frames]).astype(bool)

        # gaze format: [T, 2] pixel coordinates (x, y) or (u, v)
        gaze = None
        if 'gaze' in f:
            gaze = np.asarray(f['gaze'][:num_frames], dtype=np.float32)

    # All outputs are in world frame.
    return imgs, cam_int, cam_ext, hand, mask, gaze


if __name__ == '__main__':
    # file_path = 'data/HOT3D/processed/00000000.hdf5'
    # save_path = 'outputs/HOT3D.mp4'
    # file_path = 'data/EgoDex/processed/00000000.hdf5'
    # save_path = 'outputs/EgoDex.mp4'
    file_path = 'data/HoloAssist/processed/00000000.hdf5'
    save_path = 'outputs/HoloAssist.mp4'

    imgs, cam_int, cam_ext, hand, mask, gaze = load_vlia_hdf5(file_path)

    vis_2d_keypoint(
        cam_img=imgs,
        cam_int=cam_int,
        cam_ext=cam_ext,
        hand=hand,
        mask=mask,
        gaze=gaze,
        save_path=save_path,
    )

    vis_3d_keypoint(
        cam_img=imgs,
        cam_ext=cam_ext,
        hand=hand,
        gaze=gaze,
        mask=mask,
    )
