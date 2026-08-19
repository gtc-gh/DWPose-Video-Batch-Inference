import os
import time
import argparse
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib import pyplot as plt

from DWPose_usage import Wholebody, DWProcessor
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore')


class VideoDataset(Dataset):
    def __init__(self, video_path, frame_extraction_ratio=0, max_length=0):
        self.nFrames = None
        self.fps = None
        self.seconds = 0
        self.video_path = video_path
        self.max_length = max_length
        self.video_H = 640
        self.video_W = 640
        self.video_C = 3
        # video will be BGR & [N, H, W, C]
        self.video = self.load_video(frame_extraction_ratio)
        # frame extraction ratio: positive: every i frames, extract 1 frame
        # negative: every frame, extract i frames

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        return self.video[idx]

    def get_seconds(self):
        return self.seconds

    def get_hwc(self):
        return self.video_H, self.video_W, self.video_C

    def get_video_frames(self):
        """
        Get() function returns the processed videos in the image form.
        That is to say, return a series of images extracted from video frames.
        :return: [N, H, W, C]
        N: the number of frames (at least fps * 5)
        H: the height of the video
        W: the width of the video
        C: the channel of the video, which equals 3
        """
        return self.video

    def get_fps(self):
        return self.fps

    def get_slice(self, st, ed):
        return self.video[max(st, 0): min(ed, self.nFrames)]

    def load_video(self, frame_extraction_ratio=0):
        """Load video and apply multi-process reading."""
        vid = cv2.VideoCapture(self.video_path)
        if not vid.isOpened():
            raise ValueError("Error opening video")

        Height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.nFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(np.round(vid.get(cv2.CAP_PROP_FPS), 0))
        if self.max_length > 0:
            self.seconds = min(np.round(self.nFrames / self.fps, 2), self.max_length)
        else:
            self.seconds = np.round(self.nFrames / self.fps, 2)
        self.max_length = self.max_length * self.fps  # convert the max seconds to max frames
        print(f"video info: N: {self.nFrames}, H: {Height}, W: {Width}, fps: {self.fps}, seconds: {self.seconds}")
        start_frame = 0
        if (self.nFrames > self.max_length) and (self.max_length > 0):
            start_frame = self.nFrames - self.max_length

        if self.nFrames > 4500:
            vid.release()
            # Define how many workers to use based on the available CPU cores
            max_workers = min(4, multiprocessing.cpu_count())  # Limit to 8 processes
            print(f"Using {max_workers} CPU cores")

            # Compute the chunk size (frames per worker)
            chunk_size = (self.nFrames - start_frame) // max_workers
            frame_ranges = [(i * chunk_size + start_frame, (i + 1) * chunk_size + start_frame - 1) for i in range(max_workers)]
            frame_ranges[-1] = (frame_ranges[-1][0], self.nFrames)  # Adjust last chunk

            # Use ProcessPoolExecutor to process each chunk in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.read_frames_chunk, start, end, frame_extraction_ratio)
                    for start, end in frame_ranges
                ]
                # Collect results from all processes
                video = [frame for future in futures for frame in future.result()]
            return video
        else:
            # no multi-process reading for short videos
            video = self.read_video(vid, start_frame, self.nFrames, frame_extraction_ratio)
            vid.release()
            return video

    def read_video(self, vid, start_frame, end_frame, frame_extraction_ratio):
        vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        video_chunk = []
        count = start_frame - 1  # index in whole video
        extracted_count = 0

        # max length limitation
        while count < end_frame:
            success, frame = vid.read()
            count += 1
            if not success:
                break

            # Apply frame extraction logic
            if frame_extraction_ratio < 0:
                # extract i frames for each 1 frame
                if extracted_count == -frame_extraction_ratio:
                    extracted_count = 0
                else:
                    extracted_count += 1
                    continue
            elif frame_extraction_ratio > 0:
                # extract 1 frame for each i frames
                if count % frame_extraction_ratio == 0:
                    continue

            frame = self.preprocess_frame(frame)
            video_chunk.append(frame)

        return video_chunk

    def read_frames_chunk(self, start_frame, end_frame, frame_extraction_ratio):
        """Reads a chunk of frames from the video."""
        vid = cv2.VideoCapture(self.video_path)
        vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        video_chunk = []
        count = start_frame - 1  # index in whole video
        cnt = -1  # index in current video clip
        extracted_count = 0

        # max length limitation
        while count < end_frame:
            success, frame = vid.read()
            count += 1
            cnt += 1
            if not success:
                break

            # convert 60 fps to 30 fps
            if self.fps > 55:
                if count % 2 == 0:
                    continue

            # Apply frame extraction logic
            if frame_extraction_ratio < 0:
                # extract i frames for each 1 frame
                if extracted_count == -frame_extraction_ratio:
                    extracted_count = 0
                else:
                    extracted_count += 1
                    continue
            elif frame_extraction_ratio > 0:
                # extract 1 frame for each i frames
                if cnt % frame_extraction_ratio == 0:
                    continue

            frame = self.preprocess_frame(frame)
            video_chunk.append(frame)

        vid.release()
        return video_chunk

    def preprocess_frame(self, frame):
        """Apply preprocessing to the frame (e.g., resize)."""
        target_size = (640, 640)
        return self.resize_with_padding(np.array(frame), target_size)

    def resize_with_padding(self, image, target_size):
        """Resize image with padding to maintain aspect ratio."""
        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        delta_w = target_size[1] - new_w
        delta_h = target_size[0] - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return padded


def demotest_save_to_video(save_path, detected_images, H, W, video_name):
    video_save_path = os.path.join(save_path, f"estimated_{video_name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    video_writer = cv2.VideoWriter(f"{video_save_path}.mp4",
                                   fourcc, 30, (W, H))

    for detected_map in detected_images:
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        video_writer.write(detected_map)

    video_writer.release()
    print(f"write to {video_save_path}.mp4")


def demotest_save_to_images(save_path, detected_images, H, W, folder_name):
    # Define the directory to save images
    output_dir = os.path.join(save_path, f"estimated_{folder_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Save each detected image as an individual file
    for idx, detected_map in enumerate(detected_images):
        # Resize the image to the specified width (W) and height (H)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # Construct the file name using the provided prefix and index
        image_path = os.path.join(output_dir, f"{idx:04d}.png")

        # Save the image
        cv2.imwrite(image_path, detected_map)
    print(f"Images saved: {output_dir}")


def video_batch_inference(video_path, batch_size_1, batch_size_2, extraction_ratio, max_length, vis_res):

    video_dataset = VideoDataset(video_path, extraction_ratio, max_length)
    video_fps = video_dataset.get_fps()
    N = len(video_dataset)
    H, W, C = video_dataset.get_hwc()

    if extraction_ratio != 0:
        print(f"resized videos: N: {N}, H: {H}, W: {W}, C: {C}")

    video_metadata = {"fps": video_fps, "frame": N,"H": H, "W": W, "channel": C, "seconds": video_dataset.get_seconds()}
    device = "cuda:0"
    dwdetector = Wholebody()
    dwprocessor = DWProcessor()

    bboxes_list = []
    detected_images = []
    pose_sequences = []
    first_flag = False

    # dataloader = DataLoader(
    #     video_dataset,
    #     batch_size=batch_size_1,
    #     shuffle=False,
    #     # num_workers=16,  # Load data using 16 subprocesses
    #     pin_memory=True  # Pin memory for faster GPU transfer
    # )

    # Only one batch size
    # for batch_idx, image_batch in enumerate(dataloader):
    #     image_batch_numpy = [img.cpu().numpy() for img in image_batch]
    #     bbox_batch = dwdetector.get_bboxes(image_batch_numpy)
    #     keypoints_list_i, score_list_i = dwdetector.get_pose_sequence(image_batch_numpy, bbox_batch)
    #     detected_images_i, pose_sequences_i = dwprocessor(image_batch_numpy, keypoints_list_i, score_list_i,
    #                                                       bbox_batch)
    #     detected_images.extend(detected_images_i)
    #
    #     if not first_flag:
    #         pose_sequences = pose_sequences_i.copy()
    #         first_flag = True
    #     else:
    #         pose_sequences = np.concatenate((pose_sequences, pose_sequences_i), axis=0)
    #
    # return detected_images, pose_sequences, video_metadata  # to make pose seq to be an array is (n, 406)


    video_images = video_dataset.get_video_frames()  # [N, H, W, C]

    for i in range(0, len(video_images), batch_size_1):
        image_batch = video_images[i:min(i + batch_size_1, len(video_images))]
        bboxes_list.extend(dwdetector.get_bboxes(image_batch))

    # Get pose sequences from bounding boxes and images (DWPose)
    for i in range(0, len(video_images), batch_size_2):
        image_batch = video_images[i:min(i + batch_size_2, len(video_images))]
        bbox_batch = bboxes_list[i:min(i + batch_size_2, len(video_images))]
        keypoints_list_i, score_list_i = dwdetector.get_pose_sequence(image_batch, bbox_batch)
        detected_images_i, pose_sequences_i = dwprocessor(image_batch, keypoints_list_i, score_list_i,
                                                          bbox_batch, vis_res)
        detected_images.extend(detected_images_i)

        if not first_flag:
            pose_sequences = pose_sequences_i.copy()
            first_flag = True
        else:
            pose_sequences = np.concatenate((pose_sequences, pose_sequences_i), axis=0)

    return detected_images, pose_sequences, bboxes_list, video_metadata  # to make pose sequen to be a array is (n, 406)


def check_cuda():
    print(f"cuda is available: {torch.cuda.is_available()}")
    print(f"number of available cuda devices: {torch.cuda.device_count()}")
    print(f"current cuda device properties: {torch.cuda.get_device_properties(torch.cuda.current_device())}")
    print(f"cuda version: {torch.version.cuda}")
    print(f"cuda current device: {torch.cuda.current_device()}")


def video_pose_estimation(video_path, save_to_video_path=None, save_to_images_path=None,
                          batch_size_1=100, batch_size_2=150,
                          extraction_ratio=0, max_length=0):
    """
    Video Pose Estimation using DWPose and YOLOX with batch inference.

    :param video_path: Path to the input video file.
    :param save_to_video_path: Directory path to save the visualized output video (None to skip).
    :param save_to_images_path: Directory path to save the visualized frames as images (None to skip).
    :param batch_size_1: Batch size for human detection (YOLOX).
    :param batch_size_2: Batch size for whole-body pose estimation (DWPose).
    :param extraction_ratio: Integer extraction ratio.
                             Positive (>1): extract 1 frame every X frames.
                             Negative (<-1): extract |X| frames every frame.
                             0: process all frames. Cannot be 1. Default is 0.
    :param max_length: Limit the maximum video duration in seconds (0 for full video).
    :return: (pose_sequences, bboxes_list, video_meta)
        - pose_sequences: np.ndarray of shape (N, 406) containing bbox and 133 whole-body keypoints per frame.
        - bboxes_list: list of length N with bounding boxes [[x1, y1, x2, y2]] per frame.
        - video_meta: dict containing video metadata ('fps', 'frame', 'H', 'W', 'channel', 'seconds').
    """

    check_cuda()
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled machine.")
        return None, None, None

    # Handle string representations of None / empty strings
    if save_to_video_path in ("", "None", "none"):
        save_to_video_path = None
    if save_to_images_path in ("", "None", "none"):
        save_to_images_path = None

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    vis_res = False
    if (save_to_video_path is not None) or (save_to_images_path is not None):
        vis_res = True

    # extraction ratio cannot be 1
    assert extraction_ratio != 1, "extraction ratio cannot be 1"

    detected_frames, pose_sequences, bboxes_list, video_meta = video_batch_inference(video_path,
                                                                                     batch_size_1, batch_size_2,
                                                                                     extraction_ratio, max_length,
                                                                                     vis_res)

    # Visualization output
    if save_to_video_path is not None:
        demotest_save_to_video(save_to_video_path, detected_frames, video_meta["H"], video_meta['W'], video_name)
    if save_to_images_path is not None:
        demotest_save_to_images(save_to_images_path, detected_frames, video_meta["H"], video_meta['W'], video_name)

    return pose_sequences, bboxes_list, video_meta


def parse_args():
    """
    Parse command-line arguments for video pose estimation batch inference.
    """
    parser = argparse.ArgumentParser(
        description="DWPose & YOLOX Video Batch Inference for 2D Whole-Body Pose Estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--video_path", "-v",
        type=str,
        default="./blob_2024-07-18_11-22-28.mov",
        help="Path to the input video file (e.g. .mp4, .mov, .avi)."
    )
    parser.add_argument(
        "--save_to_video_path", "-sv",
        type=str,
        default="./",
        help="Directory path to save the visualized output video (e.g. './output/'). Set to '' or 'None' to skip."
    )
    parser.add_argument(
        "--save_to_images_path", "-si",
        type=str,
        default=None,
        help="Directory path to save the visualized individual frames (e.g. './output/frames/'). Set to '' or 'None' to skip."
    )
    parser.add_argument(
        "--batch_size_1", "-b1",
        type=int,
        default=185,
        help="Batch size for human detector (YOLOX). Recommended: 600 (A100-80G), 185 (RTX4090-24G), 115 (T4-16G)."
    )
    parser.add_argument(
        "--batch_size_2", "-b2",
        type=int,
        default=375,
        help="Batch size for whole-body pose estimator (DWPose). Recommended: 1220 (A100-80G), 375 (RTX4090-24G), 240 (T4-16G)."
    )
    parser.add_argument(
        "--extraction_ratio", "-r",
        type=int,
        default=0,
        help="Frame extraction ratio: positive (>1) extracts 1 every X frames, negative (<-1) extracts |X| frames per frame, 0 processes all frames. Cannot be 1."
    )
    parser.add_argument(
        "--max_length", "-m",
        type=float,
        default=0.0,
        help="Maximum video duration to process in seconds (0 means process full video)."
    )
    return parser.parse_args()


if __name__ == '__main__':
    """
    =============================================================================
    USAGE INSTRUCTIONS & DEMOSTRATION
    =============================================================================

    1. Command-Line Interface (CLI) Execution:
       ---------------------------------------
       # Basic inference with default parameters:
       python video_pose_estimation.py --video_path ./input.mp4

       # Save both estimated video and extracted frame images:
       python video_pose_estimation.py \\
           --video_path ./input.mp4 \\
           --save_to_video_path ./output/ \\
           --save_to_images_path ./output/frames/

       # High-throughput batch inference optimized for NVIDIA RTX 4090 (24GB VRAM):
       python video_pose_estimation.py -v ./input.mp4 -sv ./output/ -b1 185 -b2 375

       # High-throughput batch inference optimized for NVIDIA A100 (80GB VRAM):
       python video_pose_estimation.py -v ./input.mp4 -sv ./output/ -b1 600 -b2 1220

       # Downsample frames (process 1 frame every 3 frames):
       python video_pose_estimation.py -v ./input.mp4 -sv ./output/ -r 3

       # Process only the first 10 seconds:
       python video_pose_estimation.py -v ./input.mp4 -sv ./output/ -m 10.0


    2. Python API Programmatic Invocation:
       ------------------------------------
       ```python
       from video_pose_estimation import video_pose_estimation

       pose_sequences, bboxes_list, video_meta = video_pose_estimation(
           video_path="./input.mp4",
           save_to_video_path="./output/",        # Directory to save estimated video (or None)
           save_to_images_path="./output/frames/", # Directory to save estimated frame images (or None)
           batch_size_1=185,                       # Detection batch size (YOLOX)
           batch_size_2=375,                       # Pose estimation batch size (DWPose)
           extraction_ratio=0,                     # Frame extraction ratio (0 for all frames)
           max_length=0                            # Maximum video duration in seconds (0 for full)
       )

       # Output structure:
       # - pose_sequences: np.ndarray of shape (N, 406)
       #     - [0:7]:   [0, 0, x1, y1, x2, y2, bbox_score]
       #     - [7:406]: 133 whole-body keypoints, each [x, y, confidence] (normalized 0~1)
       # - bboxes_list:    list of length N containing [[x1, y1, x2, y2]] in absolute pixels
       # - video_meta:     dict {'fps', 'frame', 'H', 'W', 'channel', 'seconds'}
       ```
    =============================================================================
    """
    args = parse_args()

    print("=" * 65)
    print(" DWPose & YOLOX Video Batch Inference ")
    print("=" * 65)
    print(f" Input Video         : {args.video_path}")
    print(f" Save Video Dir      : {args.save_to_video_path}")
    print(f" Save Images Dir     : {args.save_to_images_path}")
    print(f" Detector Batch Size : {args.batch_size_1}")
    print(f" Pose Batch Size     : {args.batch_size_2}")
    print(f" Extraction Ratio    : {args.extraction_ratio}")
    print(f" Max Length (s)      : {args.max_length if args.max_length > 0 else 'Full Video'}")
    print("=" * 65)

    if not os.path.exists(args.video_path):
        print(f"\n[Demo Warning] Video file '{args.video_path}' does not exist.")
        print("Please provide a valid video file via the --video_path argument.")
        print("\nExample CLI usage:")
        print("  python video_pose_estimation.py --video_path /path/to/your/video.mp4 --save_to_video_path ./output/")
        print("\nExample Python API usage:")
        print("  from video_pose_estimation import video_pose_estimation")
        print("  poses, bboxes, meta = video_pose_estimation('/path/to/your/video.mp4', save_to_video_path='./output/')\n")
    else:
        start_time = time.time()
        pose_sequences, bboxes_list, video_meta = video_pose_estimation(
            video_path=args.video_path,
            save_to_video_path=args.save_to_video_path,
            save_to_images_path=args.save_to_images_path,
            batch_size_1=args.batch_size_1,
            batch_size_2=args.batch_size_2,
            extraction_ratio=args.extraction_ratio,
            max_length=args.max_length
        )
        elapsed_time = time.time() - start_time

        if pose_sequences is not None:
            print("\n" + "=" * 65)
            print(" Batch Inference Completed Successfully! ")
            print("=" * 65)
            print(f" Processed Frames    : {video_meta.get('frame', len(pose_sequences))}")
            print(f" Video Resolution    : {video_meta.get('W')} x {video_meta.get('H')}")
            print(f" Video FPS           : {video_meta.get('fps')}")
            print(f" Video Duration      : {video_meta.get('seconds')}s")
            print(f" Pose Sequence Shape : {pose_sequences.shape if hasattr(pose_sequences, 'shape') else len(pose_sequences)}")
            print(f" Bounding Boxes Count: {len(bboxes_list)}")
            print(f" Total Processing Time: {elapsed_time:.2f} seconds")
            print("=" * 65)

