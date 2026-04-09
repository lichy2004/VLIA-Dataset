import sys
sys.path.append('.')

import h5py
import numpy as np

from src.utils.data_format import image_decoding
from src.utils.joints import HAND_KEYPOINT_NAMES
from src.utils.visualize import vis_3d_keypoint, vis_2d_keypoint


def load_vlia_hdf5(file_path):
    """Load VLIA hdf5 and return global camera/hand data."""
    with h5py.File(file_path, 'r') as f:
        encoded_images = f['images'][:]
        imgs = np.stack([image_decoding(b, to_rgb=True) for b in encoded_images], axis=0)

        cam_int = np.asarray(f['camera']['intrinsic'][:], dtype=np.float32)
        cam_ext = np.asarray(f['camera']['extrinsic'][:], dtype=np.float32)

        keypoint_grp = f['hand']['keypoint']
        hand_keypoint = {
            side: {name: np.asarray(keypoint_grp[side][name][:], dtype=np.float32) for name in HAND_KEYPOINT_NAMES}
            for side in ('left', 'right')
        }

    hand_tfs = {
        'camera': cam_ext,
        'left': hand_keypoint['left'],
        'right': hand_keypoint['right'],
    }

    return imgs, cam_int, cam_ext, hand_keypoint, hand_tfs


if __name__ == '__main__':
    file_path = 'data/EgoDex/processed/00000000.hdf5'
    imgs, cam_int, cam_ext, hand_keypoint, hand_tfs = load_vlia_hdf5(file_path)

    vis_2d_keypoint(
        cam_img=imgs,
        cam_int=cam_int,
        hand_keypoint=hand_keypoint,
        save_path='outputs/vis_2d_keypoint.mp4',
    )

    vis_3d_keypoint(
        cam_img=imgs,
        cam_ext=cam_ext,
        hand_tfs=hand_tfs,
    )
