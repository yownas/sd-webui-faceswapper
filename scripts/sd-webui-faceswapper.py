import os
import sys
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
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision import transforms
import random
import warnings

FACE_ANALYSER = None
FACE_SWAPPER = None

class Script(scripts.Script):
    def __init__(self): # pylint: disable=useless-super-delegation
        super().__init__()

    class Rules(object):
        rules = ''
        verbose = False
        switch = False
        pass

    def LOG(self, text):
        print(f"Face swapper: {text}")

    def ERROR(self, text):
        print(f"Face swapper ERROR: {text}")

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
                swap_rules = gr.Textbox(label="Swap rules", placeholder="Example: \"1>1,3; 2>4\" or \"match gender age\"", lines=1)

        return [is_enabled, replace, restore, source_face, swap_rules]

    def get_face_swapper(self):
        global FACE_SWAPPER
        if FACE_SWAPPER is None:
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128.onnx')
            sys.stdout = open(os.devnull, 'w')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
            sys.stdout = sys.__stdout__
        return FACE_SWAPPER

    def get_face_analyser(self):
        global FACE_ANALYSER
        if FACE_ANALYSER is None:
            sys.stdout = open(os.devnull, 'w')
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l')
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
            sys.stdout = sys.__stdout__
        return FACE_ANALYSER

    def swap_matchrules(self, img, target_img, in_faces, out_faces, swaprules):
        if len(in_faces) and len(out_faces):
            rridx = 0
            for out_idx in range(len(out_faces)):
                idx = None
                if shared.state.interrupted:
                    break
                if 'age' in swaprules.rules:
                    gap = sys.maxsize
                    for i in range(len(in_faces)):
                        diff = abs(out_faces[out_idx].age - in_faces[i].age)
                        if diff < gap:
                            gap = diff
                            idx = i
                    if swaprules.verbose:
                        self.LOG(f"Match: age gap {gap}, faces {idx+1}>{out_idx+1}")
                elif 'similar' in swaprules.rules: # Match similarity
                    gap = -1
                    for i in range(len(in_faces)):
                        f1 = out_faces[out_idx].normed_embedding
                        f2 = in_faces[i].normed_embedding
                        # Based on https://learnopencv.com/face-recognition-with-arcface/
                        sim = max(np.dot(f1, f2) / (np.sqrt(np.dot(f1, f1)) * np.sqrt(np.dot(f2, f2))), 0)
                        if sim > gap:
                            gap = sim
                            idx = i
                    if swaprules.verbose:
                        self.LOG(f"Match: similar gap {gap}, faces {idx+1}>{out_idx+1}")
                elif 'ssim' in swaprules.rules: # Match ssim (very very experimental)
                    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0)
                    gap = 0
                    for i in range(len(in_faces)):
                        ofb = out_faces[out_idx].bbox
                        ifb = in_faces[i].bbox
                        t = transforms.Compose([transforms.ToTensor()])
                        f1 = t(Image.fromarray(img).crop((ofb[0], ofb[1], ofb[2], ofb[3])).resize((128, 128))).unsqueeze(0)
                        f2 = t(Image.fromarray(target_img).crop((ifb[0], ifb[1], ifb[2], ifb[3])).resize((128, 128))).unsqueeze(0)

                        sim = float(SSIM(f1, f2))
                        if sim > gap:
                            gap = sim
                            idx = i
                    if swaprules.verbose:
                        self.LOG(f"Match: SSIM gap {gap}, faces {idx+1}>{out_idx+1}")
                elif 'random' in swaprules.rules: # Match randomly
                    idx = random.randrange(0, len(in_faces))
                    if swaprules.verbose:
                        self.LOG(f"Match: random, faces {idx+1}>{out_idx+1}")
                else:
                    idx = rridx%len(in_faces)
                    rridx+=1

                if idx is not None:
                    if swaprules.verbose:
                        self.LOG(f"Match: swap {out_idx+1} with {idx+1}")
                    if out_idx <= len(out_faces) and idx <= len(in_faces):
                        img = self.get_face_swapper().get(img, out_faces[out_idx], in_faces[idx], paste_back=True)
                    else:
                        self.ERROR(f"Index out of range: {idx} > {len(faces)} or {in_face} > {len(targets)}")
                if shared.opts.live_previews_enable:
                    shared.state.assign_current_image(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        return(img)

    # run at the end of sequence for always-visible scripts
    def postprocess(self, p, processed, is_enabled, replace, restore, source_face_dict, swap_rules):  # pylint: disable=arguments-differ
        if is_enabled and not shared.state.interrupted:
            warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
            swaprules = self.Rules()

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
                if img_len:
                    swap_rules = re.sub(r'[\s;:|]+', r' ', swap_rules)
                    swap_rules, cnt = re.subn(r' *verbose *', ' ', swap_rules)
                    if cnt:
                        self.LOG("Verbose mode enabled")
                        swaprules.verbose = True
                    swap_rules, cnt = re.subn(r' *switch *', ' ', swap_rules)
                    if cnt:
                        if swaprules.verbose:
                            self.LOG("Switch images")
                        swaprules.switch = True
                        orig_target_img = target_img
                        orig_targets = targets
                    swap_rules = re.sub(r'(similar|same|like)', r'similar', swap_rules)
                    swap_rules = re.sub(r'(sex|gender)', r'sex', swap_rules)
                    swap_rules = re.sub(r'(age|old)', r'age', swap_rules)
                    swap_rules = re.sub(r'(<|=)', r'>', swap_rules)
                    swap_rules = re.sub(r' *([>,]) *', r'\1', swap_rules) # Remove \s around > & ,
                    swap_rules = re.sub(r' +', r' ', swap_rules)
                    swap_rules = re.sub(r'(^ | $)', r'', swap_rules) # Trim
                    if not swap_rules:
                        swap_rules = '*>*' # default rule
                    swaprules.rules = swap_rules

                for i in range(img_len):
                    if shared.state.interrupted:
                        break
                    tgt=0
                    try:
                        img = cv2.cvtColor(np.asarray(processed.images[i]), cv2.COLOR_RGB2BGR)
                        if shared.opts.live_previews_enable:
                            shared.state.assign_current_image(processed.images[i])
                        faces = sorted(self.get_face_analyser().get(img), key=lambda x: x.bbox[0])
                        if swaprules.switch:
                            target_img = img
                            targets = faces
                            img = orig_target_img
                            faces = orig_targets
                        if faces:
                            if 'match' in swaprules.rules: # "match sex age"
                                in_f, in_m, out_f, out_m = [], [], [], []

                                # sort faces by sex
                                for face in targets:
                                    if "sex" in swaprules.rules and face.sex == "F":
                                        in_f.append(face)
                                    else:
                                        in_m.append(face)
                                for face in faces:
                                    if "sex" in swaprules.rules and face.sex == "F":
                                        out_f.append(face)
                                    else:
                                        out_m.append(face)

                                if swaprules.verbose:
                                    self.LOG(f"Rules: {swaprules.rules}")
                                # Swap faces
                                img = self.swap_matchrules(img, target_img, in_f, out_f, swaprules)
                                img = self.swap_matchrules(img, target_img, in_m, out_m, swaprules)
                            else:
                                # Use swap rules
                                allout = ",".join(map(str, list(range(1, len(faces)+1))))
                                alltgt = list(range(1, len(targets)+1))

                                for rule in swaprules.rules.split(' '):
                                    in_faces, out_faces = rule.split('>', 1)

                                    # Make list of input faces
                                    inlist = []
                                    inidx = 0
                                    for in_face in in_faces.split(','):
                                        inlist += alltgt if in_face == '*' else [int(in_face)]

                                    # replace * with all out_faces
                                    out_faces = re.sub('\*', allout, out_faces)

                                    for out_idx in out_faces.split(','):
                                        idx=int(out_idx)
                                        in_face = inlist[inidx%len(inlist)]
                                        inidx+=1
                                        if swaprules.verbose:
                                            self.LOG(f"Swap {idx} with {in_face}")
                                        if idx <= len(faces) and in_face <= len(targets):
                                            img = self.get_face_swapper().get(img, faces[idx-1], targets[in_face-1], paste_back=True)
                                        else:
                                            self.ERROR(f"Index out of range: {idx} > {len(faces)} or {in_face} > {len(targets)}")
                                        if shared.opts.live_previews_enable:
                                            shared.state.assign_current_image(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

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
