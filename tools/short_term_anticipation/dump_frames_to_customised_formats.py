from argparse import ArgumentParser
from pathlib import Path
from typing import List
from collections import defaultdict
import itertools
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from ego4d_forecasting.datasets.short_term_anticipation import (
    PyAVVideoReader,
    Ego4DHLMDB,
)
import cv2
from PIL import Image


parser = ArgumentParser()

parser.add_argument("path_to_annotations", type=Path)
parser.add_argument("path_to_videos", type=Path)
parser.add_argument("path_to_output_lmdbs", type=Path)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--context_frames", type=int, default=32)
parser.add_argument(
    "--fname_format", type=str, default="{video_id:s}_{frame_number:07d}"
)
parser.add_argument("--frame_height", type=int, default=320)
parser.add_argument("--video_uid", type=str, default=None)

parser.add_argument("--save_as_video", action="store_true")
parser.add_argument("--save_as_film", action="store_true")
parser.add_argument("--stride", type=int, default=1)

args = parser.parse_args()


class PyAVSTADataset(Dataset):
    def __init__(
        self,
        annotations,
        path_to_videos,
        existing_keys,
        fps=30,
        max_chunk_size=32,
        retry=10,
    ):
        print(
            "Sampling from {} annotations with a temporal context of {} seconds".format(
                len(annotations), args.context_frames / fps
            )
        )
        existing_frames = defaultdict(list)
        for k in existing_keys:
            video_id, frame_number = k.decode().split("_")
            existing_frames[video_id].append(int(frame_number))

        self.path_to_videos = path_to_videos
        self.retry = retry
        if args.video_uid is not None:
            annotations = [a for a in annotations if a["video_uid"] in args.video_uid]

        frames_per_video = defaultdict(list)

        for ann in annotations:
            video_id = ann["video_id"]
            last_frame = ann["frame"]
            first_frame = np.max([0, last_frame - args.context_frames + 1])
            frame_numbers = np.arange(last_frame, first_frame-1, -args.stride)
            frames_per_video[video_id].append(frame_numbers)

        self.chunks = []

        for k, frame_chunks in frames_per_video.items():
            if args.save_as_video or args.save_as_film:
                for chunk in frame_chunks:
                    self.chunks.append((k, np.sort(np.unique(chunk))))
            else:
                raise NotImplementedError

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        video_id, frame_numbers = self.chunks[index]

        frames = {}

        for i in range(self.retry):
            frame_numbers = np.setdiff1d(frame_numbers, list(frames.keys()))
            vr = PyAVVideoReader(
                str(self.path_to_videos / f"{video_id}.mp4"), height=args.frame_height
            )
            imgs = vr[frame_numbers]

            added = 0
            for f, img in zip(frame_numbers, imgs):
                if img is not None:
                    frames[f] = img
                    added += 1

            if added == len(frame_numbers) or i == (self.retry - 1):
                keys = [
                    args.fname_format.format(video_id=video_id, frame_number=f)
                    for f in frames.keys()
                ]
                ims = list(frames.values())

                missing_frames = np.setdiff1d(frame_numbers, list(frames.keys()))

                if len(missing_frames) > 0:
                    print(
                        f"WARNING: could not read the following frames from {video_id}:",
                        ", ".join([str(x) for x in missing_frames]),
                    )

                return ims, keys


def collate(batch):
    frames = [sample[0] for sample in batch]
    keys = [sample[1] for sample in batch]
    frames = list(itertools.chain.from_iterable(frames))
    keys = list(itertools.chain.from_iterable(keys))

    return frames, keys


def _save_mp4(output_filename: Path, images: List[np.ndarray]):
    output_dir = output_filename.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Define the output video filename and codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Define the frame width and height, assuming all images have the same dimensions
    frame_width, frame_height = images[0].shape[1], images[0].shape[0]
    print("frame_width, frame_height:", frame_width, frame_height)

    # Define the frame rate in frames per second (FPS)
    fps = 32 / args.stride

    # Initialize the video writer object
    video_writer = cv2.VideoWriter(
        str(output_filename), fourcc, fps, (frame_width, frame_height)
    )

    # Write each image as a frame to the video
    for i, image in enumerate(images):
        # save each image to output_dir in png format
        cv2.imwrite(str(output_dir / f"{i:06d}.png"), image)
        video_writer.write(image)

    # Release the video writer object
    video_writer.release()

def _dump_sample(frames, keys):
    video_id = keys[-1].split("_")[0]
    idx = np.where([k.startswith(video_id) for k in keys])[0]
    these_frames = [frames[i] for i in idx]
    if args.save_as_video:  # and len(these_frames)>100:
        path_to_output_videos = (
            Path(args.path_to_output_lmdbs.parent) / "videos"
        )
        _save_mp4(
            path_to_output_videos / (keys[-1] + ".mp4"),
            these_frames
        )
    elif args.save_as_film:
        path_to_output_films = (
            Path(args.path_to_output_lmdbs.parent) / f"films_s{args.stride}_f{args.context_frames}" / video_id
        )
        path_to_output_films.mkdir(parents=True, exist_ok=True)
        # Create a new image with the combined width and height
        film = Image.new("RGB", (these_frames[0].shape[1], args.frame_height*args.context_frames//args.stride))
        # film = Image.new("RGB", (these_frames[0].shape[1]*args.context_frames, args.frame_height))
        # x_offset = 0
        for i, frame in enumerate(these_frames, start=0):
            pil_img = Image.fromarray(np.uint8(frame[:,:,::-1]))
            # Paste the images onto the new image
            # film.paste(pil_img, (x_offset, 0))
            film.paste(pil_img, (0, i*args.frame_height))
            # x_offset += pil_img.size[0]
        # Save the new image
        film.save(path_to_output_films / (f"{keys[-1]}.jpg"))
    else:
        raise NotImplementedError

def main():
    train = json.load(open(args.path_to_annotations / "fho_sta_train.json"))
    val = json.load(open(args.path_to_annotations / "fho_sta_val.json"))
    test = json.load(open(args.path_to_annotations / "fho_sta_test_unannotated.json"))

    ## Merge all annotations
    annotations = []
    for j in [train, val, test]:
        annotations += j["annotations"]

    ## Define the dataset and dataloader
    dset = PyAVSTADataset(
        annotations,
        args.path_to_videos,
        existing_keys=[],
        # existing_keys=l.get_existing_keys(),
        max_chunk_size=args.context_frames,
    )
    dloader = DataLoader(
        dset, batch_size=args.batch_size, collate_fn=collate, num_workers=8
    )

    for frames, keys in tqdm(dloader):
        _dump_sample(frames, keys)


if __name__ == "__main__":
    main()
    exit(0)
