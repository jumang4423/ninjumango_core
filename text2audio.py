from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os
import random
from riffusion.cli import image_to_audio
import json
from datetime import datetime
import sys

def load_json(json_filepath):
    with open(json_filepath) as json_file:
        data = json.load(json_file)
        model_id = data["model_id"]
        audio_out_dir = data["audio_out_dir"]
        hugging_face_token = data["hugging_face_token"]
        prompts = data["prompts"]
        negative_prompts = data["negative_prompts"]

    return model_id, audio_out_dir, hugging_face_token, prompts, negative_prompts

def out_json(json_filepath, audio_dir):
    # { audio_dir: ["/content/hoge.wav", "/content/fuga.wav"] }
    data = {
        "audio_dirs": audio_dir
    }
    with open(json_filepath, 'w') as outfile:
        json.dump(data, outfile)

arg1 = sys.argv[1]
arg2 = sys.argv[2]
model_id, audio_out_dir, hugging_face_token, prompts, negative_prompts = load_json(arg1)

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
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_auth_token=hugging_face_token,
    scheduler=scheduler
)

height = 512
width = 512
# if no audio out folder, make
if os.path.exists(audio_out_dir) == False:
    os.mkdir(audio_out_dir)

def getDayStr():
  now = datetime.now()
  return now.strftime("%y%m%d%H%M%S")

audio_file_dir = []

for i, prompt in enumerate(prompts):
    image = pipe(
        prompt,
        negative_prompts=negative_prompts[i],
        num_inference_steps=25,
        guidance_scale=7.5,
        height=height, width=width,
    ).images[0]

    img_dir = "/content/output" + str(i) + ".png"
    image.save(img_dir)
    randomNumber = random.randint(0, 1000)
    audio_out_filepath=audio_out_dir + "/" + getDayStr() + "_" + prompts[i].replace(" ", "_") + str(randomNumber) + ".wav"
    image_to_audio(image=img_dir, audio=audio_out_filepath)
    audio_file_dir.append(audio_out_filepath)

out_json(arg2, audio_file_dir)
