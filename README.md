# sd-webui-faceswapper
Extension to swap faces in Stable Diffusion webui

# Installation
Install manually.

It might break, and not work with some other extensions who need another version of the insightface package. This extensions is experimental, I can not help you.

Requires you to download inswapper_128.onnx and place it in the sd-webui-faceswapper folder.

This extension is inspired by [Roop](https://github.com/s0md3v/roop), you can find the inswapper_128.onnx in their wiki. I recommend that you use the extension by the same developer. [sd-webui-roop](https://github.com/s0md3v/sd-webui-roop)

# Usage
Check the `Enable Script` checkbox and upload an image with a face, generate as usual. `Replace original` will overwrite the original image instead of keeping it. `Restore faces` will use the Webui's builting restore faces, trying to make things look better.

The target image may contain multiple faces. These will be selected round-robin, from left to right, when replacing faces in the image.

You can also use Gradios sketch tool to mask faces you do not want to include.

# API
Simple example using [sdwebuiapi](https://github.com/mix1009/sdwebuiapi).

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
                    'args': [True, True, True, {'image': face, 'mask': None}],
                }
            }
        )

result1.image.show()
```

# Disclaimer
It pains me that this part is needed. I do not condone or encourage ANY malicious use of this extension. It is intented as a fun experiment and a tool to get consistant characters. Do not use it with any real people without their explicit concent. Realize that misuse might come with actual legal consequences, do not be a creepy idiot.
