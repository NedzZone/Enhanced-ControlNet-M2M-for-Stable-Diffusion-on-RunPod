import copy
import os
import shutil
import cv2
import gradio as gr
import modules.scripts as scripts
from modules import images
from modules.processing import process_images
from modules.shared import opts
from PIL import Image
import numpy as np

_BASEDIR = "/controlnet-m2m"
_BASEFILE = "animation"

def get_all_frames(video_path):
    if video_path is None:
        return None
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if ret:
            frame_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break
    return frame_list

def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image

def get_min_frame_num(video_list):
    min_frame_num = -1
    for video in video_list:
        if video is None:
            continue
        else:
            frame_num = len(video)
            if min_frame_num < 0:
                min_frame_num = frame_num
            elif frame_num < min_frame_num:
                min_frame_num = frame_num
    return min_frame_num

def save_gif(path, image_list, name, duration):
    tmp_dir = path + "/tmp/"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    for i, image in enumerate(image_list):
        images.save_image(image, tmp_dir, f"output_{i}")
    os.makedirs(f"{path}{_BASEDIR}", exist_ok=True)
    image_list[0].save(f"{path}{_BASEDIR}/{name}.gif", save_all=True, append_images=image_list[1:], optimize=False, duration=duration, loop=0)

class Script(scripts.Script):  
    def title(self):
        return "controlnet m2m"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        ctrls_group = ()
        max_models = opts.data.get("control_net_max_models_num", 1)

        with gr.Group():
            with gr.Accordion("ControlNet-M2M", open=False):
                duration = gr.Slider(label="Duration", value=50.0, minimum=10.0, maximum=200.0, step=10, interactive=True, elem_id='controlnet_movie2movie_duration_slider')
                with gr.Tabs():
                    for i in range(max_models):
                        with gr.Tab(f"ControlNet-{i}"):
                            with gr.TabItem("Movie Input"):
                                ctrls_group += (gr.Textbox(label="Video File Path", placeholder="/path/to/your/video.mp4", elem_id=f"video_path_{i}"),)
                            with gr.TabItem("Image Input"):
                                ctrls_group += (gr.Image(source='upload', brush_radius=20, mirror_webcam=False, type='numpy', tool='sketch', elem_id=f'image_{i}'),)
                            ctrls_group += (gr.Checkbox(label="Save preprocessed", value=False, elem_id=f"save_pre_{i}"),)
                
                ctrls_group += (duration,)
            return ctrls_group

    def run(self, p, *args):
        try:
            contents_num = opts.data.get("control_net_max_models_num", 1)
            arg_num = 3
            item_list = []
            video_list = []

            for input_set in [args[i:i + arg_num] for i in range(0, len(args), arg_num)]:
                video_path = input_set[0]
                if video_path:
                    video_frames = get_all_frames(video_path)
                    if video_frames:
                        item_list.append([video_frames, "video"])
                        video_list.append(video_frames)
                    else:
                        print(f"Failed to load video from path: {video_path}")

                if len(input_set) > 1 and input_set[1] is not None:
                    item_list.append([cv2.cvtColor(pil2cv(input_set[1]["image"]), cv2.COLOR_BGRA2RGB), "image"])

            save_pre = args[2:contents_num * arg_num:arg_num] if len(args) >= contents_num * arg_num else []
            duration = args[contents_num * arg_num] if len(args) > contents_num * arg_num else 50

            frame_num = get_min_frame_num(video_list)
            if frame_num > 0:
                output_image_list = []
                pre_output_image_list = [[] for _ in range(contents_num)]

                for frame in range(frame_num):
                    copy_p = copy.copy(p)
                    copy_p.control_net_input_image = [item[0][frame] if item[1] == "video" else item[0] for item in item_list]

                    proc = process_images(copy_p)
                    output_image_list.append(proc.images[0])

                    for i, save in enumerate(save_pre):
                        if save:
                            try:
                                pre_output_image_list[i].append(proc.images[i + 1])
                            except IndexError:
                                print(f"proc.images[{i}] failed")

                    copy_p.close()

                seq = images.get_next_sequence_number(f"{p.outpath_samples}{_BASEDIR}", "")
                filename = f"{seq:05}-{proc.seed}-{_BASEFILE}"
                save_gif(p.outpath_samples, output_image_list, filename, duration)
                proc.images = [f"{p.outpath_samples}{_BASEDIR}/{filename}.gif"]

                for i, save in enumerate(save_pre):
                    if save:
                        save_gif(p.outpath_samples, pre_output_image_list[i], f"{filename}-control{i}", duration)
                        proc.images.append(f"{p.outpath_samples}{_BASEDIR}/{filename}-control{i}.gif")
            else:
                print("No frames found in the video(s), running normal image processing")
                proc = process_images(p)

        except Exception as e:
            print(f"Error in script: {e}")
            raise

        return proc
