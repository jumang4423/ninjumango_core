from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os
import random
import sys

model_name = sys.argv[1]


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
    model_name,
    torch_dtype=torch.float16,
    use_auth_token=hugging_face_token,
    scheduler=scheduler
).to("cuda")
