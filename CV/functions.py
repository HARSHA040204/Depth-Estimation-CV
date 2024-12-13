import cv2
import torch
import zipfile
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

def load_midas():
    """Load the MiDaS model and transformation pipeline."""
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to("cpu")
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    return midas, transforms.small_transform


def depth_estimation(image, midas, transform):
    """Perform depth estimation on a given image."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to("cpu")

    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))  # Normalize
    depth_map = (depth_map * 255).astype(np.uint8)
    return depth_map


import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image

def apply_strict_colormap(depth_map):
    """Apply a custom colormap with near as purple and far as yellow-orange."""
    # Normalize depth map to range [0, 1] and invert for proper mapping
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_map_normalized = 1 - depth_map_normalized  # Invert for correct color mapping


    custom_colors = [
    "#60283f",  # Near (Dark Red-Purple)
    "#60283f",  # Near (Dark Red-Purple)
    "#60283f",  # Near (Dark Red-Purple)
    "#642584",  # Near (Deep Purple)
    "#642584",  # Near (Deep Purple)
    "#642584",  # Near (Deep Purple)
    "#4f4dbf",  # Mid-range (Medium Purple-Blue)
    "#4f4dbf",  # Mid-range (Medium Purple-Blue)
    
    
    "#0766f2", 
    "#2bb8cb",  # Mid-range (Cyan Blue)
    
    "#1aa2d1",  # Far-mid range (Light Cyan)
    # "#76d7f2",  # Far (Soft Light Cyan)
    
    "#aeeffb",  # Far (Very Light Cyan)
    
    # "#d4f8fc"   # Far (Almost White Cyan) 
    ]


    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_purple_to_yellow", custom_colors)

    # Apply the custom colormap to the normalized depth map
    depth_colormap = custom_cmap(depth_map_normalized)  # Get RGBA values

    # Convert to RGB (8-bit format)
    depth_colormap_rgb = (depth_colormap[:, :, :3] * 255).astype(np.uint8)

    return depth_colormap_rgb






def apply_bw_colormap(depth_map):
    """Apply a simple black and white colormap to the depth map."""
    # Normalize depth map to range [0, 255]
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_map_bw = (depth_map_normalized * 255).astype(np.uint8)
    return depth_map_bw


def process_single_image_color(uploaded_file, midas, transform):
    """Process a single image for color depth estimation."""
    image = np.array(Image.open(uploaded_file))
    depth_map = depth_estimation(image, midas, transform)
    depth_map_color = apply_strict_colormap(depth_map)

    output_path = "output_depth_color.png"
    cv2.imwrite(output_path, depth_map_color)  # Save as image
    return output_path


def process_single_image_bw(uploaded_file, midas, transform):
    """Process a single image for black and white depth estimation."""
    image = np.array(Image.open(uploaded_file))
    depth_map = depth_estimation(image, midas, transform)
    depth_map_bw = apply_bw_colormap(depth_map)

    output_path = "output_depth_bw.png"
    cv2.imwrite(output_path, depth_map_bw)  # Save as image
    return output_path


def process_zip_file_color(uploaded_zip, midas, transform):
    """Process a ZIP file containing images for color depth estimation, including nested folders."""
    zip_path = "uploaded_dataset.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())

    output_dir = "processed_dataset_color"
    os.makedirs(output_dir, exist_ok=True)

    # Unzip the uploaded file
    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name in z.namelist():
            if file_name.endswith(("jpg", "jpeg", "png")):
                # Check the parent folder structure
                folder_name = os.path.dirname(file_name)
                
                # Create a corresponding "Depth" folder if it doesn't exist
                depth_folder = os.path.join(output_dir, folder_name, "Depth")
                os.makedirs(depth_folder, exist_ok=True)

                with z.open(file_name) as img_file:
                    img = Image.open(img_file)
                    image = np.array(img)
                    depth_map = depth_estimation(image, midas, transform)
                    depth_map_color = apply_strict_colormap(depth_map)

                    # Save the depth map with the same name as the original image
                    output_path = os.path.join(depth_folder, os.path.basename(file_name))
                    cv2.imwrite(output_path, depth_map_color)

    # Zip the processed depth maps
    output_zip_path = "processed_color_dataset.zip"
    with zipfile.ZipFile(output_zip_path, "w") as z:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

    return output_zip_path


def process_zip_file_bw(uploaded_zip, midas, transform):
    """Process a ZIP file containing images for black and white depth estimation, including nested folders."""
    zip_path = "uploaded_dataset.zip"
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())

    output_dir = "processed_dataset_bw"
    os.makedirs(output_dir, exist_ok=True)

    # Unzip the uploaded file
    with zipfile.ZipFile(zip_path, "r") as z:
        for file_name in z.namelist():
            if file_name.endswith(("jpg", "jpeg", "png")):
                # Check the parent folder structure
                folder_name = os.path.dirname(file_name)
                
                # Create a corresponding "Depth" folder if it doesn't exist
                depth_folder = os.path.join(output_dir, folder_name, "Depth")
                os.makedirs(depth_folder, exist_ok=True)

                with z.open(file_name) as img_file:
                    img = Image.open(img_file)
                    image = np.array(img)
                    depth_map = depth_estimation(image, midas, transform)
                    depth_map_bw = apply_bw_colormap(depth_map)

                    # Save the depth map with the same name as the original image
                    output_path = os.path.join(depth_folder, os.path.basename(file_name))
                    cv2.imwrite(output_path, depth_map_bw)

    # Zip the processed depth maps
    output_zip_path = "processed_bw_dataset.zip"
    with zipfile.ZipFile(output_zip_path, "w") as z:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                z.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

    return output_zip_path




def process_video_color(uploaded_video, midas, transform):
    """Process a video file for color depth estimation."""
    video_path = uploaded_video.name
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(video_path)
    output_path = "processed_video_color.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        depth_map = depth_estimation(frame, midas, transform)
        depth_map_color = apply_strict_colormap(depth_map)
        out.write(depth_map_color)

    cap.release()
    out.release()
    return output_path


def process_video_bw(uploaded_video, midas, transform):
    """Process a video file for black and white depth estimation."""
    video_path = uploaded_video.name
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(video_path)
    output_path = "processed_video_bw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Generate the depth map and apply the black and white colormap
        depth_map = depth_estimation(frame, midas, transform)
        depth_map_bw = apply_bw_colormap(depth_map)

        # Convert grayscale (1 channel) to RGB (3 channels)
        depth_map_bw_rgb = cv2.cvtColor(depth_map_bw, cv2.COLOR_GRAY2RGB)

        # Write the frame to the video file
        out.write(depth_map_bw_rgb)

    cap.release()
    out.release()
    return output_path
