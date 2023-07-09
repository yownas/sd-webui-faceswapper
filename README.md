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

There are two ways to specify the rules. Either `target face(s)>generated face number(s)`, for example `2>1,3` will take the second face from the left in the image you uploaded and place it on the first and third face in the generated image. Multiple rules can be added, separated with a space or newline (or one of `:;|`). Using `*` will mean any face. For example `1,3,5>*` to place face number 1, 3 and 5 from the uploaded image to all face in the generated image. If it runs out of faces defined in the left-hand side it will start from the beginning.

Or you can let the script automatically try to match gender and/or age by using the keyword `match` and any combination of `gender` and/or one of `age` or `similar`. Please note that this is a bit unreliable and the model seem to struggle with gender, especially with very young or old people. `age` will try to match the persons age while `similar` will try to match facial feature/shape (and often hilariously fails).

You can also use `match random` to randomly choose a face. If you enter multiple options, the order of precedence is; `age`, `similar` and last `random`.

Adding keyword `verbose` will enable some logging.

Adding keyword `switch` will swap the uploaded image with the generated before swapping faces and will place generated faces on the image you uploaded.

# Settings and Face swapper tab

In the Webui `Settings` you can enable a Face swapper tab. This part is still under development and will get more functions in the future.

At the moment you can upload two images and swap faces between them. Or use the sketch tool to mark faces to be swapped.

Saved images should show up in `outputs/save/`.

# Examples

No rule, using the default `*>*`. All images in the uploaded swapped to all in the generated.
![default_example](https://github.com/yownas/sd-webui-faceswapper/assets/13150150/03f0a631-82a9-4f2c-ad47-18f361ee9473)

Multiple faces in uploaded image. Swap with Hermione and then John Wick, until all faces are done.
![def](https://github.com/yownas/sd-webui-faceswapper/assets/13150150/c8b8daa2-4ba1-4cfa-b06d-873dc1d583ad)

Masking one of the faces.
![mask](https://github.com/yownas/sd-webui-faceswapper/assets/13150150/cba79db4-8323-4f7f-90b2-b829adaf7374)

`1>1,2,3 2>5,6,7` Hermione to face 1, 2 and 3, John Wick to face 5, 6 and 7.
![1to123-2to567](https://github.com/yownas/sd-webui-faceswapper/assets/13150150/0ed2ac51-652c-4a47-983a-8be8b37de2ab)

`2,1>3,4,5` Place face #2 and #1 on faces 3 to 5.
![21to345](https://github.com/yownas/sd-webui-faceswapper/assets/13150150/2dd25ea7-3714-45d2-a921-730dfe9f04a7)

`match gender` Trying to match the gender of the faces replaced.
![matchgender](https://github.com/yownas/sd-webui-faceswapper/assets/13150150/88df8872-4be5-438a-8387-0d692cabd17c)

`match gender switch`, using the keyword `switch` to switch the images before swapping faces.
![matchgenderswitch](https://github.com/yownas/sd-webui-faceswapper/assets/13150150/1bdb2fcd-d695-4753-b88f-30fb33ccb660)



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
