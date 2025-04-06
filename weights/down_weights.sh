# Please download the following weights under this folder
mkdir -p weights
cd weights

# Depth Anything V2 Large
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true
echo "Depth Anything V2 Large downloaded"

# SAM2 Hiera Large
wget https://huggingface.co/facebook/sam2-hiera-large/resolve/main/sam2_hiera_large.pt?download=true
echo "SAM2 Hiera Large downloaded"

# SV3D
wget https://huggingface.co/camenduru/sv3d/resolve/main/sv3d_p.safetensors?download=true
echo "SV3D downloaded"

cd ..
