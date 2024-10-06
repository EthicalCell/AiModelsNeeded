echo "ComfyUI Install"
#
# Install 
#
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
apt-get install python3.10-venv
python -m venv venv
echo "activate venv"
source comfyvenv/bin/activate
which pip
which python
pip install ipykernel
python -m ipykernel install --name=comfyvenv
#git pull
pip install --upgrade pip
pip install scikit-image
pip install imageio-ffmpeg
pip install segment-anything
pip install xformers!=0.0.18 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://download.pytorch.org/whl/cu117
echo "Nodes and Models Install" 
#
# Install Nodes and Models
#
### SDXL
### I recommend these workflow examples: https://comfyanonymous.github.io/ComfyUI_examples/sdxl/

#wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors -P ./models/checkpoints/

# SDXL ReVision
#wget -c https://huggingface.co/comfyanonymous/clip_vision_g/resolve/main/clip_vision_g.safetensors -P ./models/clip_vision/

# SD1.5
#wget -c https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors -P ./models/checkpoints/
wget -c "https://civitai.com/api/download/models/132760?token=2dba7b4fd93f0bd0a9b4641fdd1a70eb" -O "./models/checkpoints/AbsoluteReality.safetensors"
wget -c "https://civitai.com/api/download/models/429454?token=2dba7b4fd93f0bd0a9b4641fdd1a70eb" -O "./models/checkpoints/EpicPhotogasm.safetensors"
wget -c "https://civitai.com/api/download/models/501240?token=2dba7b4fd93f0bd0a9b4641fdd1a70eb" -O "./models/checkpoints/RealisticVision.safetensors"
wget -c "https://civitai.com/api/download/models/143906?token=2dba7b4fd93f0bd0a9b4641fdd1a70eb" -O "./models/checkpoints/epiCRealism.safetensors"
wget -c "https://civitai.com/api/download/models/646523?token=2dba7b4fd93f0bd0a9b4641fdd1a70eb" -O "./models/checkpoints/LeoSam.safetensors"
#w/ Noise Select

# SD2
#wget -c https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors -P ./models/checkpoints/

# Flux
#mkdir ./models/checkpoints/unet
#wget -c https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors -P ./models/checkpoints/unet

# Some SD1.5 anime style
#wget -c https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix2/AbyssOrangeMix2_hard.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix3/AOM3A1_orangemixs.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/Models/AbyssOrangeMix3/AOM3A3_orangemixs.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/Linaqruf/anything-v3.0/resolve/main/anything-v3-fp16-pruned.safetensors -P ./models/checkpoints/

# Waifu Diffusion 1.5 (anime style SD2.x 768-v)
#wget -c https://huggingface.co/waifu-diffusion/wd-1-5-beta3/resolve/main/wd-illusion-fp16.safetensors -P ./models/checkpoints/


# unCLIP models
#wget -c https://huggingface.co/comfyanonymous/illuminatiDiffusionV1_v11_unCLIP/resolve/main/illuminatiDiffusionV1_v11-unclip-h-fp16.safetensors -P ./models/checkpoints/
#wget -c https://huggingface.co/comfyanonymous/wd-1.5-beta2_unCLIP/resolve/main/wd-1-5-beta2-aesthetic-unclip-h-fp16.safetensors -P ./models/checkpoints/

#CLIP models
wget -c https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.safetensors -P ./models/clip/

#AnimateDiff models
wget -c https://huggingface.co/CiaraRowles/TemporalDiff/resolve/main/temporaldiff-v1-animatediff.safetensors -P ./models/animatediff_models/
wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt -P ./models/animatediff_models/

# VAE
wget -c https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -P ./models/vae/
#wget -c https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/VAEs/orangemix.vae.pt -P ./models/vae/
#wget -c https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt -P ./models/vae/


# Loras
#wget -c https://civitai.com/api/download/models/10350 -O ./models/loras/theovercomer8sContrastFix_sd21768.safetensors #theovercomer8sContrastFix SD2.x 768-v
#wget -c https://civitai.com/api/download/models/10638 -O ./models/loras/theovercomer8sContrastFix_sd15.safetensors #theovercomer8sContrastFix SD1.x
#wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors -P ./models/loras/ #SDXL offset noise lora
# clone my Loras
git clone https://github.com/EthicalCell/AiModelsNeeded.git ./models/loras/

# T2I-Adapter
#wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_depth_sd14v1.pth -P ./models/controlnet/
#wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_seg_sd14v1.pth -P ./models/controlnet/
#wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_sketch_sd14v1.pth -P ./models/controlnet/
#wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_keypose_sd14v1.pth -P ./models/controlnet/
wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_openpose_sd14v1.pth -P ./models/controlnet/
#wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_color_sd14v1.pth -P ./models/controlnet/
#wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_canny_sd14v1.pth -P ./models/controlnet/

# T2I Styles Model
#wget -c https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_style_sd14v1.pth -P ./models/style_models/

# CLIPVision model (needed for styles model)
#wget -c https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin -O ./models/clip_vision/clip_vit14.bin


# ControlNet
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_scribble_fp16.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_seg_fp16.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11u_sd15_tile_fp16.safetensors -P ./models/controlnet/

# ControlNet SDXL
#wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors -P ./models/controlnet/
#wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors -P ./models/controlnet/

# Controlnet Preprocessor nodes by Fannovel16
#cd custom_nodes && git clone https://github.com/Fannovel16/comfy_controlnet_preprocessors; cd comfy_controlnet_preprocessors && python install.py


# GLIGEN
#wget -c https://huggingface.co/comfyanonymous/GLIGEN_pruned_safetensors/resolve/main/gligen_sd14_textbox_pruned_fp16.safetensors -P ./models/gligen/


# ESRGAN upscale model
wget -c https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./models/upscale_models/
#wget -c https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth -P ./models/upscale_models/
#wget -c https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth -P ./models/upscale_models/
wget -c "https://civitai.com/api/download/models/158264?token=2dba7b4fd93f0bd0a9b4641fdd1a70eb" -O "./models/upscale_models/8xnmkd-faces160000g-upscaler.pth"
wget -c "https://civitai.com/api/download/models/156841?token=2dba7b4fd93f0bd0a9b4641fdd1a70eb" -O "./models/upscale_models/4x_NMKD_Superscale.pth"

# ComfyUI Manager
git clone https://github.com/ltdrdata/ComfyUI-Manager ./custom_nodes/ComfyUI-Manager
git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes ./custom_nodes/ComfyUI_Comfyroll_CustomNodes

#install controlnet
git clone https://github.com/Fannovel16/comfyui_controlnet_aux ./custom_nodes/comfyui_controlnet_aux
pip install -r ./custom_nodes/comfyui_controlnet_aux/requirements.txt

git clone https://github.com/cubiq/ComfyUI_essentials ./custom_nodes/ComfyUI_essentials
git clone https://github.com/shiimizu/ComfyUI_smZNodes ./custom_nodes/ComfyUI_smZNodes
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale ./custom_nodes --recursive
git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved ./custom_nodes/ComfyUI-AnimateDiff-Evolved
git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts ./custom_nodes/ComfyUI-Custom-Scripts

#install florence2
git clone https://github.com/kijai/ComfyUI-Florence2 ./custom_nodes/ComfyUI-Florence2
pip install -r ./custom_nodes/ComfyUI-Florence2/requirements.txt

#install frame interpolation
git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation ./custom_nodes/ComfyUI-Frame-Interpolation
python ./custom_nodes/ComfyUI-Frame-Interpolation/install.py

#install ComfyUI-Impact Pack
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack ./custom_nodes/ComfyUI-Impact-Pack
pip install -r ./custom_nodes/ComfyUI-Impact-Pack/requirements.txt
python install.py

#install Inspire Pack
git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack ./custom_nodes/ComfyUI-Inspire-Pack
pip install -r ./custom_nodes/ComfyUI-Inspire-Pack/requirements.txt

#Install KJ Noes
git clone https://github.com/kijai/ComfyUI-KJNodes ./custom_nodes/ComfyUI-KJNodes
pip install -r ./custom_nodes/ComfyUI-KJNodes/requirements.txt

git clone https://github.com/storyicon/comfyui_segment_anything ./custom_nodes/comfyui_segment_anything
pip install -r ./custom_nodes/comfyui_segment_anything/requirements.txt

git clone 'https://github.com/kijai/ComfyUI-segment-anything-2' ./custom_nodes/comfyui-segment-anything-2
pip install -r ./custom_nodes/comfyui-segment-anything-2/requirements.txt

git clone https://github.com/un-seen/comfyui-tensorops ./custom_nodes/comfyui-tensorops
pip install -r ./custom_nodes/comfyui-tensorops/requirements.txt

git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite ./custom_nodes/ComfyUI-VideoHelperSuite
pip install -r ./custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt

git clone https://github.com/nicofdga/DZ-FaceDetailer ./custom_nodes/DZ-FaceDetailer
pip install -r ./custom_nodes/DZ-FaceDetailer/requirements.txt

git clone https://github.com/mav-rik/facerestore_cf ./custom_nodes/facerestore_cf
pip install -r ./custom_nodes/facerestore_cf/requirements.txt

git clone https://github.com/marhensa/sdxl-recommended-res-calc ./custom_nodes/sdxl-recommended-res-calc

sudo apt-get install libmagickwand-dev
git clone https://github.com/Fannovel16/ComfyUI-MagickWand ./custom_nodes/ComfyUI-MagickWand
pip install -r ./custom_nodes/ComfyUI-MagickWand/requirements.txt

echo "Done!"