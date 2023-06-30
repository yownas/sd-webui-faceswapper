# sd-webui-faceswapper
Extension to swap faces in Stable Diffusion webui

# Installation
Install manually.

On Windows you will also need to download and install [Visual Studio](https://visualstudio.microsoft.com/downloads/). Make sure to include the Python and C++ packages.

It might break, and not work with some other extensions who need another version of the insightface package. This extensions is experimental, I can not help you.

Requires you to download [inswapper_128.onnx](https://huggingface.co/henryruhs/roop/resolve/main/inswapper_128.onnx) and place it in the sd-webui-faceswapper folder.

This extension is inspired by [Roop](https://github.com/s0md3v/roop), you can find the inswapper_128.onnx in their wiki. I recommend that you use the extension by the same developer. [sd-webui-roop](https://github.com/s0md3v/sd-webui-roop)

# Usage
Check the `Enable Script` checkbox and upload an image with a face, generate as usual. `Replace original` will overwrite the original image instead of keeping it. `Restore faces` will use the Webui's builting restore faces, trying to make things look better.

The target image may contain multiple faces. These will be selected round-robin, from left to right, when replacing faces in the image.

You can also use Gradios sketch tool to mask faces you do not want to include.

At the bottom there is a textbox for "swap rules" which will let you decide which faces are places where.

The format is `(target face)>(generated face number(s))`, for example `2>1,3` will take the second face from the left in the image you uploaded and place it on the first and third face in the generated image. Multiple rules can be added, separated with a space or newline (or one of `:;|`). Using `*` will mean any face in the generated image that doesn't have a rule for it, or a face selected round-robin from the uploaded image. One special case is if you don't want any special rules for the generated faces, you can either use `*>*` (which is the default rule if nothing is specified) or multiple faces, for example `1,3,5>*` to place face number 1, 3 and 5 from the uploaded image to any face in the generated image.

# API
Simple example using [sdwebuiapi](https://github.com/mix1009/sdwebuiapi).

Args are: Enabled, Replace original, Restore faces, the input image and mask (can be None), and the Swap rule string (can be None).

```
import webuiapi
import base64

api = webuiapi.WebUIApi(host='localhost', port=7860)

with open("face.jpg", "rb") as img_file:
    face = base64.b64encode(img_file.read()).decode('utf-8')

result1 = api.txt2img(
        prompt="photo of a cool superhero, funny hair",
        negative_prompt="mask, nsfw",
        steps=25,
        width=512,
        height=768,
        sampler_index="Euler a",
        cfg_scale=7,
        seed=-1,
        alwayson_scripts = {
            'face swapper': {
                    'args': [True, True, True, {'image': face, 'mask': None}, None],
                }
            }
        )

result1.image.show()
```

# Disclaimer
It pains me that this part is needed. I do not condone or encourage ANY malicious use of this extension. It is intented as a fun experiment and a tool to get consistant characters. Do not use it with any real people without their explicit concent. Realize that misuse might come with actual legal consequences, do not be a creepy idiot.
