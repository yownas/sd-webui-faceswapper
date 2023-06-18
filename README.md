# sd-webui-faceswapper
Extension to swap faces in Stable Diffusion webui

# Installation
Install manually.

It might break, and not work with some other extensions who need another version of the insightface package. This extensions is experimental, I can not help you.

Requires you to download inswapper_128.onnx and place it in the sd-webui-faceswapper folder.

This extension is inspired by [Roop](https://github.com/s0md3v/roop), you can find the inswapper_128.onnx in their wiki. I recommend that you use the extension by the same developer. [sd-webui-roop](https://github.com/s0md3v/sd-webui-roop)

# Usage
Check the `Enable Script` checkbox and upload an image with a face, generate as usual. `Replace original` will overwrite the original image instead of keeping it. `Restore faces` will use the Webui's builting restore faces, trying to make things look better.

The target image may contain multiple faces. These will be selected round-robbin, from left to right, when replacing faces in the image.

# Disclaimer
It pains me that this part is needed. I do not condone or encourage ANY malicious use of this extension. It is intented as a fun experiment and a tool to get consistant characters. Do not use it with any real people without their explicit concent. Realize that misuse might come with actual legal consequences, do not be a creepy idiot.
