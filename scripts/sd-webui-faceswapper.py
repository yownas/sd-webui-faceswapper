import os
import cv2
import torch
import gradio as gr
import insightface
import onnxruntime
from PIL import Image
import numpy as np
from tqdm import tqdm
from modules.api.api import decode_base64_to_image
from modules import scripts, shared, face_restoration
import re

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
                source_face = gr.Image(label="Face", tool="sketch", type="numpy")
            with gr.Row():
                swap_rules = gr.Textbox(label="Swap rules", lines=1)

        return [is_enabled, replace, restore, source_face, swap_rules]

    def get_face_swapper(self):
        global FACE_SWAPPER
        if FACE_SWAPPER is None:
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
        return FACE_SWAPPER

    def get_face_analyser(self):
        global FACE_ANALYSER
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l')
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
        return FACE_ANALYSER

    # run at the end of sequence for always-visible scripts
    def postprocess(self, p, processed, is_enabled, replace, restore, source_face_dict, swap_rules):  # pylint: disable=arguments-differ
        if is_enabled:
            if isinstance(source_face_dict["image"], str):
                source_face = decode_base64_to_image(source_face_dict["image"])
            else:
                source_face = source_face_dict["image"]
            if isinstance(source_face_dict["mask"], str):
                source_face_mask = decode_base64_to_image(source_face_dict["mask"])
            else:
                source_face_mask = source_face_dict["mask"]

            target_img = cv2.cvtColor(np.asarray(source_face), cv2.COLOR_RGB2BGR)
            if source_face_mask is not None:
                target_mask = cv2.cvtColor(np.asarray(source_face_mask), cv2.COLOR_RGB2BGR)
                target_img = cv2.bitwise_and(target_img, cv2.bitwise_not(target_mask))

            try:
                targets = sorted(self.get_face_analyser().get(target_img), key=lambda x: x.bbox[0])
            except IndexError:
                # No face?
                return None
            img_len = len(processed.images)
            with tqdm(total=img_len, desc="Face swapping", unit="image") as progress:
                for i in range(img_len):
                    tgt=0
                    try:
                        img = cv2.cvtColor(np.asarray(processed.images[i]), cv2.COLOR_RGB2BGR)
                        faces = sorted(self.get_face_analyser().get(img), key=lambda x: x.bbox[0])
                        if faces:
                            if not swap_rules:
                                swap_rules = '*>*' # default rule
                            swap_rules = re.sub(r'[\s;:|]+', r' ', swap_rules)
                            swap_rules = re.sub(r' *([>,]) *', r'\1', swap_rules) # Remove \s around > & ,
                            swap_rules = re.sub(r' +', r' ', swap_rules)
                            swap_rules = re.sub(r'(^ | $)', r'', swap_rules) # Trim

                            swap_pairs = {}
                            rr_targets = list(range(len(targets)))
                            for rule in swap_rules.split(' '):
                                in_face, out_faces = rule.split('>', 1)
                                if out_faces == '*':
                                  if in_face == '*':
                                      rr_targets = list(range(len(targets)))
                                  else:
                                      rr_targets = list(map(int, in_face.split(',')))
                                else:
                                    for out_face in out_faces.split(','):
                                        swap_pairs[out_face] = -1 if in_face == '*' else int(in_face) 

                            rr = 1
                            for idx in range(1, len(faces)+1):
                                idx_s = str(idx)
                                in_face = swap_pairs[idx_s] if idx_s in swap_pairs else swap_pairs['*'] if '*' in swap_pairs else -1
                                if in_face == -1: # round-robin
                                    in_face = rr_targets[rr%len(rr_targets)]
                                    rr+=1
                                if in_face is not None:
                                    img = self.get_face_swapper().get(img, faces[idx-1], targets[in_face-1], paste_back=True)
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
