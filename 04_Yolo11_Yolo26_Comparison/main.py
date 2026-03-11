
from glob import glob
from video_grid import VideoGrid # create_video_grid

if __name__ == "__main__":
    # Use glob to create a list of the desired input videos
    input_paths = \
        glob("outputs/out_*_onnx_*.mp4") + \
        glob("outputs/out_*_pt_*2.mp4") + \
        ["outputs/out_yolo26n_pt_video2.mp4"]
    num_videos = len(input_paths)
    output_path = f"outputs/out_all_{num_videos}.avi"

    # Create an instance of the class
    vg = VideoGrid (input_paths, output_path)
    vg.create_video_grid ()

