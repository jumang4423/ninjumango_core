from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionImg2ImgPipeline
import torch
import os
import random
from riffusion.cli import image_to_audio, audio_to_image
import json
from datetime import datetime
import sys
import requests
from PIL import Image
from io import BytesIO


def load_json(json_filepath):
    with open(json_filepath) as json_file:
        data = json.load(json_file)
        model_id = data["model_id"]
        audio_out_dir = data["audio_out_dir"]
        hugging_face_token = data["hugging_face_token"]
        audio_file = data["audio_file"]
        prompt = data["prompt"]
        negative_prompt = data["negative_prompt"]

    return model_id, audio_out_dir, hugging_face_token, audio_file, prompt, negative_prompt

def out_json(json_filepath, audio_dir):
    data = {
        "audio_dir": audio_dir
    }
    with open(json_filepath, 'w') as outfile:
        json.dump(data, outfile)

arg1 = sys.argv[1]
arg2 = sys.argv[2]
model_id, audio_out_dir, hugging_face_token, audio_file, prompt, negative_prompt = load_json(arg1)

scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    trained_betas=None,
    predict_epsilon=True,
    thresholding=False,
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    lower_order_final=True,
)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_auth_token=hugging_face_token,
    scheduler=scheduler
)
pipe = pipe.to("cuda")

height = 512
width = 512
# if no audio out folder, make
if os.path.exists(audio_out_dir) == False:
    os.mkdir(audio_out_dir)

def getDayStr():
  now = datetime.now()
  return now.strftime("%y%m%d%H%M%S")

response = requests.get(audio_file)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image.thumbnail((512, 512))
image = pipe(
    prompt,
    negative_prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    height=height, width=width,
    image=init_image, strength=0.75
).images[0]
img_dir = "/content/tmp.png"
image.save(img_dir)
randomNumber = random.randint(0, 1000)
audio_out_dir=audio_out_dir + "/" + getDayStr() + "_" + Path(audio_file).stem + "_" + prompt.replace(" ", "_") + str(randomNumber) + ".wav"
image_to_audio(image=img_dir, audio=audio_out_filepath)
out_json(arg2, audio_out_dir)
