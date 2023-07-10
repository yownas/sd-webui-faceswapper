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
from modules import scripts, script_callbacks, shared, face_restoration, images, generation_parameters_copypaste
import re
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision import transforms
import random
import warnings

FACE_ANALYSER = None
FACE_SWAPPER = None

class Sd_webui_faceswap(scripts.Script):
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
                source_face = gr.Image(show_label=False, tool="sketch", type="numpy")
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

#-----------------------------------------------#
# Face swap tab & functions

#FIXME should probably be shared with the code above
def get_face_swapper():
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128.onnx')
        sys.stdout = open(os.devnull, 'w')
        FACE_SWAPPER = insightface.model_zoo.get_model(model_path, download=False, download_zip=False)
        sys.stdout = sys.__stdout__
    return FACE_SWAPPER

def get_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        sys.stdout = open(os.devnull, 'w')
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l')
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
        sys.stdout = sys.__stdout__
    return FACE_ANALYSER

def select_faces(face_img, mask_img):
    allfaces = sorted(get_face_analyser().get(face_img), key=lambda x: x.bbox[0])
    faces = []
    for face in allfaces:
        #b = face.bbox
        b = list(map(lambda x: max(0, int(x)), face.bbox))
        mask = mask_img[b[1]:b[3], b[0]:b[2]]
        if mask.size and np.amax(mask) > 0:
            faces.append(face)
    if len(faces) == 0:
        faces = allfaces
    return faces

def faceswap_l2r(image_l, image_r):
    img_l = cv2.cvtColor(np.asarray(image_l['image']), cv2.COLOR_RGB2BGR)
    mask_l = cv2.cvtColor(np.asarray(image_l['mask']), cv2.COLOR_RGB2GRAY)
    try:
        faces_l = select_faces(img_l, mask_l)
    except IndexError:
        # No face?
        return image_r

    img_r = cv2.cvtColor(np.asarray(image_r['image']), cv2.COLOR_RGB2BGR)
    mask_r = cv2.cvtColor(np.asarray(image_r['mask']), cv2.COLOR_RGB2GRAY)
    try:
        faces_r = select_faces(img_r, mask_r)
    except IndexError:
        # No face?
        return image_r

    if faces_l and faces_r:
        idx = 0
        for out_face in faces_r:
            img_r = get_face_swapper().get(img_r, out_face, faces_l[idx%len(faces_l)], paste_back=True)
            idx+=1

    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    return img_r

def faceswap_r2l(image_l, image_r):
    return faceswap_l2r(image_r, image_l)

def faceswap_save(image):
    images.save_image(Image.fromarray(image['image']), shared.opts.outdir_save, "", prompt="faceswapper", extension="png")

def faceswap_copy(image):
    return image

def add_tab():
    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row():
            with gr.Column(scale=1):
                image_l = gr.Image(elem_id='faceswapper_left', show_label=False, interactive=True, tool='sketch')
            with gr.Column(scale=1):
                    image_r = gr.Image(elem_id='faceswapper_right', show_label=False, interactive=True, tool='sketch')
            with gr.Column(scale=1):
                    result_img = gr.Image(elem_id='faceswapper_result', label='Result', show_label=True, interactive=False, type='pil', tool=None)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    copy_l = gr.Button('Copy result')
                    swap_l2r = gr.Button('Swap ->', variant='primary')
            with gr.Column(scale=1):
                with gr.Row():
                    swap_r2l = gr.Button('<- Swap', variant='primary')
                    copy_r = gr.Button('Copy result')
            with gr.Column(scale=1):
                with gr.Row():
                    send_to_buttons = generation_parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])
                    save_result = gr.Button('Save', variant='primary')

        swap_l2r.click(faceswap_l2r, show_progress=True, inputs=[image_l, image_r], outputs=[result_img])
        swap_r2l.click(faceswap_r2l, show_progress=True, inputs=[image_l, image_r], outputs=[result_img])

        copy_l.click(faceswap_copy, show_progress=False, inputs=[result_img], outputs=[image_l])
        copy_r.click(faceswap_copy, show_progress=False, inputs=[result_img], outputs=[image_r])

        save_result.click(faceswap_save, show_progress=False, inputs=[result_img], outputs=[])

        try:
            for tabname, button in send_to_buttons.items():
                generation_parameters_copypaste.register_paste_params_button(generation_parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_image_component=result_img))
        except:
            pass

    return [(tab, "Face swapper", "faceswapper")]

def on_ui_settings():
    section = ("faceswapper", "Face swapper")
    shared.opts.add_option(
        "sd_webui_faceswapper_showtab",
        shared.OptionInfo(False, "Enable Face swapper tab (requires complete restart)", section=section),
    )

if shared.opts.data.get("sd_webui_faceswapper_showtab", False):
    script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_ui_settings(on_ui_settings)
