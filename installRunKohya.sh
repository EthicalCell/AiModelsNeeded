git clone --recursive https://github.com/bmaltais/kohya_ss.git
cd kohya_ss
bash setup-runpod.sh
#sdxl vae
wget -c https://huggingface.co/stabilityai/sdxl-vae/resolve/main/config.json -P /workspace/ComfyUI/models/vae
wget -c https://huggingface.co/stabilityai/sdxl-vae/resolve/main/diffusion_pytorch_model.bin -P /workspace/ComfyUI/models/vae
wget -c https://huggingface.co/stabilityai/sdxl-vae/resolve/main/diffusion_pytorch_model.safetensors - P /workspace/ComfyUI/models/vae
wget -c https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors -P /workspace/ComfyUI/models/vae
bash gui.sh --share --headless