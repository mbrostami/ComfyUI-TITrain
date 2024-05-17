import subprocess
import time
import comfy.utils
import folder_paths
import numpy as np
from PIL import Image
import os
import sys 
import pexpect
from comfy import model_management
import latent_preview

def run_script(args: list, total_steps: int) -> str:
    child = pexpect.spawn(args[0], args[1:], encoding='utf-8', timeout=None)
    progress_bar = comfy.utils.ProgressBar(total_steps)

    stdout = ""
    # callback = latent_preview.prepare_callback(model, total_steps)
    final_output = ""
    try:
        while True:
            line = child.readline()
            if line:
                stdout += line
                if line.strip().find('-- STEP') != -1: # if -- STEP found in the output, increase the step
                    # callback(step, line.replace('-- STEP', ''), 0, total_steps)
                    progress_bar.update(1)  # Update progress bar
                final_output = line.strip()
                print(final_output)
            else:
                # no more output
                if child.eof():
                    break
    except pexpect.EOF:
        print("Process finished.")
    except pexpect.TIMEOUT:
        print("Timeout occurred while waiting for output.")

    child.close()
    stdout += child.before 

    # Handle errors
    if child.exitstatus != 0:
        error_message = f"Script error with exit status {child.exitstatus}: {final_output}"
        raise RuntimeError(error_message)

    return stdout

def save_images(images, subfolder="train", filename_prefix="Train_ComfyUI"):
    output_dir = folder_paths.get_output_directory()
    compress_level = 4
    full_output_folder, filename, counter, _, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
    os.makedirs(os.path.join(full_output_folder, subfolder), exist_ok=True)
    for (batch_number, image) in enumerate(images):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = None
        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_with_batch_num}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, subfolder, file), pnginfo=metadata, compress_level=compress_level)
        counter += 1

    return os.path.join(full_output_folder, subfolder)

class TextualInversionTrainingSDXL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"forceInput": True}),
                "train_dir": ("STRING", {
                    "multiline": False,
                    "default": "train"
                }),
                "pretrained_model_name": ("STRING", { # folder_paths.get_filename_list("checkpoints"), 
                    "multiline": False,
                    "default": "stabilityai/stable-diffusion-xl-base-1.0"
                }),
                "learnable_property": ("STRING", {
                    "multiline": False,
                    "default": "object"
                }),
                "placeholder_token": ("STRING", {
                    "multiline": False,
                    "default": "cat-toy"
                }),
                "embedding_name": ("STRING", {
                    "multiline": False,
                    "default": "cat-toy"
                }),
                "initializer_token": ("STRING", {
                    "multiline": False,
                    "default": "toy"
                }),
                "resolution": ("STRING", {
                    "multiline": False,
                    "default": "768"
                }),
                "max_train_steps": ("INT", {
                    "default": 500, 
                    "min": 1,
                    "max": 8096,
                    "step": 64,
                    "display": "number"
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0005, 
                    "min": 0,
                    "step": 0.00001, 
                    "display": "number"
                }),
                "mixed_precision": (["no", "fp16", "bf16"], {
                    "default": "no"
                }),
                "validation_steps": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "max": 2048,
                    "step": 1,
                    "display": "number"
                }),
                "num_vectors": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "max": 15,
                    "step": 1,
                    "display": "number"
                }),
                "batch_size": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": 4, 
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "display": "number"
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt_extra": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "train"

    OUTPUT_NODE = True

    CATEGORY = "train"
    def train(
            self, 
            images, 
            train_dir, 
            pretrained_model_name, 
            learnable_property, 
            placeholder_token,
            embedding_name,
            initializer_token,
            resolution,
            max_train_steps,
            learning_rate,
            mixed_precision,
            validation_steps,
            num_vectors,
            batch_size,
            gradient_accumulation_steps,
            seed,
            prompt_extra,
    ):
        # free memory
        loadedmodels=model_management.current_loaded_models
        unloaded_model = False
        for i in range(len(loadedmodels) -1, -1, -1):
            m = loadedmodels.pop(i)
            m.model_unload()
            del m
            unloaded_model = True
        if unloaded_model:
            model_management.soft_empty_cache()
        input_path = save_images(images, subfolder=train_dir)
        # if pretrained_model_name in folder_paths.get_filename_list("checkpoints"):
        #     pretrained_model_path = folder_paths.get_full_path("checkpoints", pretrained_model_name)
        # else: 
        pretrained_model_path = pretrained_model_name
            
        base_path = f'{os.path.dirname(__file__)}/textual_inversion_sdxl.py'
        args = [
            'accelerate', 'launch', base_path, 
            f'--pretrained_model_name_or_path={pretrained_model_path}',
            f'--train_data_dir={input_path}',
            f'--learnable_property={learnable_property}',
            f'--placeholder_token=<{placeholder_token}>',
            f'--initializer_token={initializer_token}',
            f'--resolution={resolution}',
            f'--train_batch_size={batch_size}',
            f'--gradient_accumulation_steps={gradient_accumulation_steps}',
            f'--max_train_steps={max_train_steps}',
            f'--learning_rate={learning_rate}',
            '--scale_lr',
            '--lr_scheduler=constant',
            '--lr_warmup_steps=0',
            f'--validation_steps={validation_steps}',
            f'--mixed_precision={mixed_precision}',
            f'--output_dir=./save/checkpoints/{placeholder_token}',
            f'--embedding_name=./models/embeddings/{embedding_name}.safetensors',
            '--resume_from_checkpoint=latest',
            f'--num_vectors={num_vectors}',
            f'--seed={seed}',
            f'--prompt_extra={prompt_extra}',
            '--enable_xformers_memory_efficient_attention',
        ]
        if validation_steps > 0:
            args.append(f'--num_validation_images=1')
            args.append(f'--validation_steps={validation_steps}')
            args.append(f'--validation_prompt={initializer_token}, a photo of <{placeholder_token}>')

        return run_script(args, max_train_steps)
        

class TextualInversionTraining:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"forceInput": True}),
                "train_dir": ("STRING", {
                    "multiline": False,
                    "default": "train"
                }),
                "pretrained_model_name": ("STRING", { # folder_paths.get_filename_list("checkpoints"), 
                    "multiline": False,
                    "default": "runwayml/stable-diffusion-v1-5"
                }),
                "learnable_property": ("STRING", {
                    "multiline": False,
                    "default": "object"
                }),
                "placeholder_token": ("STRING", {
                    "multiline": False,
                    "default": "cat-toy"
                }),
                "embedding_name": ("STRING", {
                    "multiline": False,
                    "default": "cat-toy"
                }),
                "initializer_token": ("STRING", {
                    "multiline": False,
                    "default": "toy"
                }),
                "resolution": ("STRING", {
                    "multiline": False,
                    "default": "512"
                }),
                "max_train_steps": ("INT", {
                    "default": 500, 
                    "min": 1,
                    "max": 8096,
                    "step": 64,
                    "display": "number"
                }),
                "learning_rate": ("FLOAT", {
                    "default": 5.0e-04, 
                    "min": 0,
                    "step": 0.00001, 
                    "display": "number"
                }),
                "mixed_precision": (["no", "fp16", "bf16"], {
                    "default": "no"
                }),
                "validation_steps": ("INT", {
                    "default": 0, 
                    "min": 0,
                    "max": 2048,
                    "step": 1,
                    "display": "number"
                }),
                "num_vectors": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "max": 15,
                    "step": 1,
                    "display": "number"
                }),
                "batch_size": ("INT", {
                    "default": 1, 
                    "min": 1,
                    "step": 1,
                    "display": "number"
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": 4, 
                    "min": 1,
                    "step": 1,
                    "display": "number"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt_extra": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "train"

    OUTPUT_NODE = True

    CATEGORY = "train"
    def train(
            self, 
            images, 
            train_dir, 
            pretrained_model_name, 
            learnable_property, 
            placeholder_token,
            embedding_name,
            initializer_token,
            resolution,
            max_train_steps,
            learning_rate,
            mixed_precision,
            validation_steps,
            num_vectors,
            batch_size,
            gradient_accumulation_steps,
            seed,
            prompt_extra,
    ):
        # free memory
        loadedmodels=model_management.current_loaded_models
        unloaded_model = False
        for i in range(len(loadedmodels) -1, -1, -1):
            m = loadedmodels.pop(i)
            m.model_unload()
            del m
            unloaded_model = True
        if unloaded_model:
            model_management.soft_empty_cache()
        input_path = save_images(images, subfolder=train_dir)
        # if pretrained_model_name in folder_paths.get_filename_list("checkpoints"):
        #     pretrained_model_path = folder_paths.get_full_path("checkpoints", pretrained_model_name)
        # else: 
        pretrained_model_path = pretrained_model_name
            
        base_path = f'{os.path.dirname(__file__)}/textual_inversion.py'
        args = [
            'accelerate', 'launch', base_path, 
            f'--pretrained_model_name_or_path={pretrained_model_path}',
            f'--train_data_dir={input_path}',
            f'--learnable_property={learnable_property}',
            f'--placeholder_token=<{placeholder_token}>',
            f'--initializer_token={initializer_token}',
            f'--resolution={resolution}',
            f'--train_batch_size={batch_size}',
            f'--gradient_accumulation_steps={gradient_accumulation_steps}',
            f'--max_train_steps={max_train_steps}',
            f'--learning_rate={learning_rate}',
            '--scale_lr',
            '--lr_scheduler=constant',
            '--lr_warmup_steps=0',
            f'--validation_steps={validation_steps}',
            f'--mixed_precision={mixed_precision}',
            f'--output_dir=./save/checkpoints/{placeholder_token}',
            f'--embedding_name=./models/embeddings/{embedding_name}.safetensors',
            '--resume_from_checkpoint=latest',
            f'--num_vectors={num_vectors}',
            f'--seed={seed}',
            f'--prompt_extra={prompt_extra}',
            '--enable_xformers_memory_efficient_attention',
        ]
        if validation_steps > 0:
            args.append(f'--num_validation_images=1')
            args.append(f'--validation_steps={validation_steps}')
            args.append(f'--validation_prompt={initializer_token}, a photo of <{placeholder_token}>')

        return run_script(args, max_train_steps)
        
NODE_CLASS_MAPPINGS = {
    "TextualInversionTrainingSDXL": TextualInversionTrainingSDXL,
    "TextualInversionTraining": TextualInversionTraining
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextualInversionTrainingSDXL": "Textual Inversion Training SDXL",
    "TextualInversionTraining": "Textual Inversion Training SD1.5"
}
