import os
import cv2
import torch
import gradio as gr
import insightface
import onnxruntime
from PIL import Image
import numpy as np
from tqdm import tqdm

from modules import scripts, shared, face_restoration

FACE_ANALYSER = None
FACE_SWAPPER = None

class Script(scripts.Script):
    def __init__(self): # pylint: disable=useless-super-delegation
        super().__init__()

    # script title to show in ui
    def title(self):
        return 'Face swapper'

    # is ui visible: process/postprocess triggers for always-visible scripts otherwise use run as entry point
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # ui components
    def ui(self, is_img2img): # pylint: disable=unused-argument
        with gr.Accordion('Face swapper', open=False):
            with gr.Row():
                is_enabled = gr.Checkbox(label='Script Enabled', value=False)
                replace = gr.Checkbox(label='Replace original', value=False)
                restore = gr.Checkbox(label='Restore faces', value=False)
            with gr.Row():
                source_face = gr.Image(label="Face")

        return [is_enabled, replace, restore, source_face]

    def get_face_swapper(self):
        global FACE_SWAPPER
        if FACE_SWAPPER is None:
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, download=True, download_zip=True)
            #FACE_SWAPPER = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
        return FACE_SWAPPER

    def get_face_analyser(self):
        global FACE_ANALYSER
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l')
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
        return FACE_ANALYSER

    # run at the end of sequence for always-visible scripts
    def postprocess(self, p, processed, is_enabled, replace, restore, source_face):  # pylint: disable=arguments-differ
        if is_enabled:
            if isinstance(source_face, str):
                from modules.api.api import decode_base64_to_image
                source_face = decode_base64_to_image(source_face)
            target_img = cv2.cvtColor(np.asarray(source_face), cv2.COLOR_RGB2BGR)
            try:
                target = sorted(self.get_face_analyser().get(target_img), key=lambda x: x.bbox[0])[0]
            except IndexError:
                # No face?
                return None
            img_len = len(processed.images)
            with tqdm(total=img_len, desc="Face swapping", unit="image") as progress:
                for i in range(img_len):
                    try:
                        img = cv2.cvtColor(np.asarray(processed.images[i]), cv2.COLOR_RGB2BGR)
                        faces = self.get_face_analyser().get(img)
                        if faces:
                            for face in faces:
                                img = self.get_face_swapper().get(img, face, target, paste_back=True)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if restore:
                            img = Image.fromarray(face_restoration.restore_faces(np.asarray(img)))
                        if replace:
                            processed.images[i] = img
                        else:
                            processed.images.append(img)
                    except Exception as e:
                        print(e)
                        pass
                    progress.update(1)
