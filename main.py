import os
import time

import jax

from image_processing.enums import NoiseType
from video_utils.dummy_video_writer import DummyVideoWriter
from video_utils.imutils_video_reader import ImutilsVideoReader

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


from image_processing.algo_image_processor import AlgoImageProcessor
from video_utils.gif_video_writer import GIFVideoWriter
from video_converter import VideoConverter

input_path = 'videos/avengers.mp4'
output_dir_path = 'output_videos/avengers_1_blue'

# video_reader = CV2VideoReader(input_path, target_pixel_count=300000, target_framerate=20)
video_reader = ImutilsVideoReader(input_path, target_pixel_count=300000, target_framerate=20, block_size=(2, 2))

video_writer = GIFVideoWriter(output_dir_path, frames_per_clip=250, target_framerate=20)
# video_writer = DummyVideoWriter()
# video_writer = CompressedVideoWriter(output_path)
image_processor = AlgoImageProcessor(NoiseType.BLUE)

batch_size = 250

video_converter = VideoConverter(video_reader, video_writer, image_processor, batch_size)

start = time.time()

# with jax.disable_jit():
video_converter.convert()

print(time.time() - start)
