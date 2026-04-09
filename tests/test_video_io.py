import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.common.video_io import (
    cleanup_video_only_outputs,
    decode_video_frames,
    encode_dataset_h264_videos,
    load_masks_by_names_with_video_fallback,
    resolve_output_policy,
)


class TestVideoIO(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="aiaa_videoio_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @unittest.skipUnless(shutil.which("ffmpeg") is not None, "ffmpeg unavailable")
    def test_encode_decode_h264_roundtrip(self) -> None:
        ds_root = self.tmpdir / "exp" / "toy"
        ds_root.mkdir(parents=True, exist_ok=True)
        policy = resolve_output_policy({"output_policy": {"video_only": True, "write_h264_videos": True}})

        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for idx in range(4):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            cv2.rectangle(frame, (8 + idx, 10), (24 + idx, 30), (10, 200, 100), -1)
            mask = np.zeros((48, 64), dtype=np.uint8)
            cv2.rectangle(mask, (10 + idx, 12), (20 + idx, 28), 255, -1)
            frames.append(frame)
            masks.append(mask)

        restored_path, mask_path = encode_dataset_h264_videos(
            dataset_root=ds_root,
            restored_frames_bgr=frames,
            masks_u8=masks,
            fps=24.0,
            output_policy=policy,
        )
        self.assertIsNotNone(restored_path)
        self.assertIsNotNone(mask_path)
        assert restored_path is not None
        assert mask_path is not None
        self.assertTrue(restored_path.exists())
        self.assertTrue(mask_path.exists())

        decoded_frames = decode_video_frames(restored_path, as_gray=False)
        decoded_masks = decode_video_frames(mask_path, as_gray=True)
        self.assertEqual(len(decoded_frames), len(frames))
        self.assertEqual(len(decoded_masks), len(masks))
        self.assertEqual(decoded_frames[0].shape[:2], frames[0].shape[:2])

    @unittest.skipUnless(shutil.which("ffmpeg") is not None, "ffmpeg unavailable")
    def test_mask_video_fallback_loader(self) -> None:
        ds_root = self.tmpdir / "exp" / "toy"
        ds_root.mkdir(parents=True, exist_ok=True)
        policy = resolve_output_policy({"output_policy": {"video_only": True, "write_h264_videos": True}})

        frames = [np.zeros((40, 40, 3), dtype=np.uint8) for _ in range(3)]
        masks: list[np.ndarray] = []
        for idx in range(3):
            m = np.zeros((40, 40), dtype=np.uint8)
            m[8 + idx : 16 + idx, 10:20] = 255
            masks.append(m)

        _, mask_path = encode_dataset_h264_videos(
            dataset_root=ds_root,
            restored_frames_bgr=frames,
            masks_u8=masks,
            fps=24.0,
            output_policy=policy,
        )
        assert mask_path is not None
        frame_names = [f"frame_{i + 1:06d}.png" for i in range(len(frames))]
        loaded, meta = load_masks_by_names_with_video_fallback(
            mask_dir=ds_root / "masks",  # does not exist
            frame_names=frame_names,
            frame_shape=(40, 40),
            mask_video_path=mask_path,
            threshold=127,
        )
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(meta.get("source"), "mask_video")
        self.assertEqual(len(loaded), len(frame_names))
        self.assertTrue(any(int((m > 0).sum()) > 0 for m in loaded))

    @unittest.skipUnless(shutil.which("ffmpeg") is not None, "ffmpeg unavailable")
    def test_encode_odd_size_yuv420p_auto_pad(self) -> None:
        ds_root = self.tmpdir / "exp" / "odd"
        ds_root.mkdir(parents=True, exist_ok=True)
        policy = resolve_output_policy({"output_policy": {"video_only": True, "write_h264_videos": True}})

        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for idx in range(3):
            frame = np.zeros((475, 844, 3), dtype=np.uint8)
            frame[:, :, 1] = 60 + idx
            mask = np.zeros((475, 844), dtype=np.uint8)
            mask[100:220, 180:360] = 255
            frames.append(frame)
            masks.append(mask)

        restored_path, _ = encode_dataset_h264_videos(
            dataset_root=ds_root,
            restored_frames_bgr=frames,
            masks_u8=masks,
            fps=24.0,
            output_policy=policy,
        )
        assert restored_path is not None
        decoded = decode_video_frames(restored_path, as_gray=False)
        self.assertEqual(len(decoded), len(frames))
        self.assertEqual(decoded[0].shape[1] % 2, 0)
        self.assertEqual(decoded[0].shape[0] % 2, 0)

    def test_cleanup_video_only_outputs(self) -> None:
        exp_root = self.tmpdir / "exp_root"
        ds_root = exp_root / "toy"
        (exp_root / "_candidates" / "E4" / "x").mkdir(parents=True, exist_ok=True)
        (ds_root / "frames").mkdir(parents=True, exist_ok=True)
        (ds_root / "masks").mkdir(parents=True, exist_ok=True)
        (ds_root / "frames" / "frame_000001.png").write_bytes(b"x")
        (ds_root / "masks" / "frame_000001.png").write_bytes(b"x")
        (ds_root / "restored_h264.mp4").write_bytes(b"fake")
        (ds_root / "mask_h264.mp4").write_bytes(b"fake")

        policy = resolve_output_policy(
            {
                "output_policy": {
                    "video_only": True,
                    "write_h264_videos": True,
                    "auto_cleanup_intermediates": True,
                }
            }
        )
        stats = cleanup_video_only_outputs(
            exp_pred_root=exp_root,
            datasets=["toy"],
            output_policy=policy,
        )
        self.assertTrue(stats.get("enabled"))
        self.assertFalse((exp_root / "_candidates").exists())
        self.assertFalse((ds_root / "frames").exists())
        self.assertFalse((ds_root / "masks").exists())
        self.assertTrue((ds_root / "restored_h264.mp4").exists())
        self.assertTrue((ds_root / "mask_h264.mp4").exists())


if __name__ == "__main__":
    unittest.main()
