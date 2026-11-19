"""The ONE image path shared by dataset conversion and real-robot rollout.

Training images and live images must go through identical processing (the policy is trained from scratch, so any
difference in channel order, framing or resizing is a distribution shift). Both sides import these functions;
nothing else in the repo should crop, resize or reorder camera images for the real robot.

    conversion: decode_recorded(jpeg_bytes)          -> RGB 720x1280  -> to_policy_image -> RGB 120x160 uint8
    rollout:    live_frame(camera.read() BGR frame)  -> RGB 720x1280  -> to_policy_image -> RGB 120x160 uint8

Recorder quirk (SimpleTrajectoryRecorder/src/simple_recorder/episodes.py): frames are flipped BGR -> RGB on
capture (line 58), then JPEG-encoded with cv2.imencode (line 118), which assumes BGR. The stored JPEGs therefore
have red/blue swapped for standard decoders (PIL etc. show the red cup as blue), while cv2.imdecode -- the exact
inverse of cv2.imencode -- returns the true RGB frame. decode_recorded relies on that.
"""
import cv2
import numpy as np

SOURCE_HW = (720, 1280)  # recorded camera resolution (H, W)
POLICY_HW = (120, 160)   # policy input (H, W); must match `image_shape` in configs/policy/real_robot/*
RECORDER_JPEG_QUALITY = 92


def decode_recorded(jpeg_bytes) -> np.ndarray:
    """A JPEG frame from a cup_multi_shelf .h5 file -> true RGB uint8 (H, W, 3). See the module docstring."""
    rgb = cv2.imdecode(np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError("cv2.imdecode failed to decode a recorded frame")
    return rgb


def live_frame(bgr: np.ndarray, jpeg_roundtrip: bool = False) -> np.ndarray:
    """A live camera frame (BGR uint8 (H, W, 3), as SimpleTrajectoryRecorder's cameras.py returns) -> true RGB.

    jpeg_roundtrip=True additionally reproduces the recorder's JPEG step (encode at quality 92, decode), so live
    pixels match the training data bit for bit. Off by default -- the quality-92 artifacts mostly vanish in the
    downsampling; it is a thing to try if the policy does worse on live frames than in offline replay.
    """
    rgb = np.ascontiguousarray(bgr[..., ::-1])  # recorder line 58
    if jpeg_roundtrip:
        ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, RECORDER_JPEG_QUALITY])  # recorder line 118
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        rgb = decode_recorded(buf.reshape(-1))
    return rgb


def to_policy_image(rgb: np.ndarray) -> np.ndarray:
    """True RGB uint8 (720, 1280, 3) -> policy image uint8 (120, 160, 3).

    4:3 center crop (keeps the middle 960 columns: drops empty wall on the left and arm body on the right, see
    the Step 0 crop review), then an area-averaging resize. The policy's own 90% crop (random in training,
    center at inference) happens inside the model, not here.
    """
    if rgb.shape != (*SOURCE_HW, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 {(*SOURCE_HW, 3)}, got {rgb.dtype} {rgb.shape}")
    h, w = SOURCE_HW
    crop_w = h * 4 // 3
    x0 = (w - crop_w) // 2
    cropped = rgb[:, x0 : x0 + crop_w]
    return cv2.resize(cropped, (POLICY_HW[1], POLICY_HW[0]), interpolation=cv2.INTER_AREA)


def to_model_input(policy_image: np.ndarray) -> np.ndarray:
    """Policy image uint8 (H, W, 3) -> float32 (3, H, W) in [0, 1], the layout and range the policy expects
    (it then applies its own min-max normalization to [-1, 1])."""
    return policy_image.transpose(2, 0, 1).astype(np.float32) / 255.0
