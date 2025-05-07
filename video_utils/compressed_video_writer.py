import cv2

from video_utils.base_video_writer import BaseVideoWriter, VideoFrameData


class CompressedVideoWriter(BaseVideoWriter):
    def __init__(self, output_path: str, target_framerate=30):
        super().__init__(target_framerate)
        self.output_path = output_path
        self.video = None

    def close(self):
        self.video.release()

    def _init_video(self, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        # fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI
        self.video = cv2.VideoWriter(self.output_path, fourcc, 30.0, (width, height))

    def _write_frame(self, frame: VideoFrameData):
        if self.video is None:
            self._init_video(frame.image.shape[1], frame.image.shape[0])

        image = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
        self.video.write(image)
