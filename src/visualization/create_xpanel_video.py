import argparse
from moviepy.editor import VideoFileClip, clips_array
from pathlib import Path
from typing import List

def main(output_video: Path, **video_files):
    if output_video.exists():
        return
    video_clips = [VideoFileClip(str(path)) for path in video_files.values()]
    final_clip = clips_array([video_clips])  # Arrange clips horizontally
    final_clip.write_videofile(str(output_video))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_video", type=Path, required=True)

    # Use nargs='*' to accept any number of arguments
    parser.add_argument("--video_files", nargs='*', action='append', required=True)

    args = parser.parse_args()

    # Convert list of lists to dictionary
    video_files = {f"{i}": Path(path) for i, path in enumerate(args.video_files[0])}
    main(args.output_video, **video_files)
