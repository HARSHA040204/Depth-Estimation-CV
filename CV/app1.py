import streamlit as st
import cv2
import torch
import zipfile
import os
import numpy as np
from PIL import Image
from functions import (
    load_midas,
    process_single_image_color,
    process_single_image_bw,
    process_zip_file_color,
    process_zip_file_bw,
    process_video_color,
    process_video_bw
)

# Set page configuration
st.set_page_config(page_title="Vision Sense", layout="wide")

# Initialize MiDaS model
midas, transform = load_midas()

# Sidebar Navigation
st.sidebar.title("Vision Sense")
page = st.sidebar.radio("Navigation", ["Home", "Depth Estimation"])

if page == "Home":
    # Home Page Content
    st.markdown(
    """
    <div style="text-align: center;">
        <h1>Welcome to Vision Sense</h1>
    </div>
    """,
    unsafe_allow_html=True
)
    st.image(r"images/image.png", caption="Vision Sense", use_column_width=True)


    st.markdown(
        """
        ### About Vision Sense
        **Vision Sense** is an web application for **depth estimation**. 
        Depth estimation is a crucial computer vision technique that enables understanding 
        the distance of objects from the camera in a scene.

        With Vision Sense, you can:
        - Generate **Color Depth Maps** and **Black & White Depth Maps**.
        - Process depth estimation for **Single images**, **Datasets (ZIP files)**, and **Videos**.
       

        #### Why Depth Estimation?
        Depth estimation is widely used in:
        - **Autonomous vehicles** to understand the environment.
        - **3D modeling** and **AR/VR applications** for immersive experiences.
        - **Medical imaging**, **robotics**, and many other fields.

        #### Features of Vision Sense:
        - Upload images, videos, or datasets in ZIP format for processing.
        - Select between **Color** or **Black & White** depth maps based on your preference.
        - Download processed results directly for further use.

        #### Get Started
        Use the sidebar to navigate to the **Depth Estimation** section and upload your files to see the of Vision Sense in action!
        """
    )

elif page == "Depth Estimation":
    # Depth Estimation Features
    map_type = st.sidebar.radio("Choose Depth Map Type", ["Color Depth Map", "Black & White Depth Map"])
    options = st.sidebar.radio("Choose Task", ["Single Image", "ZIP File (Dataset)", "Video"])

    if map_type == "Color Depth Map":
        if options == "Single Image":
            st.title("Color Depth Estimation - Single Image")
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                output_path = process_single_image_color(uploaded_file, midas, transform)

                # Show the depth map
                st.image(output_path, caption="Color Depth Map", use_column_width=True)

                # Provide a download button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Color Depth Map",
                        data=file,
                        file_name="color_depth_map.png",
                        mime="image/png"
                    )

        elif options == "ZIP File (Dataset)":
            st.title("Color Depth Estimation - ZIP File (Dataset)")
            uploaded_zip = st.file_uploader("Upload a ZIP File", type=["zip"])

            if uploaded_zip:
                output_zip_path = process_zip_file_color(uploaded_zip, midas, transform)

                # Provide a download button
                with open(output_zip_path, "rb") as file:
                    st.download_button(
                        label="Download Processed ZIP",
                        data=file,
                        file_name="processed_color_dataset.zip",
                        mime="application/zip"
                    )

        elif options == "Video":
            st.title("Color Depth Estimation - Video")
            uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

            if uploaded_video:
                st.video(uploaded_video)
                output_video_path = process_video_color(uploaded_video, midas, transform)

                # Provide a download button
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="processed_video_color.mp4",
                        mime="video/mp4"
                    )

    elif map_type == "Black & White Depth Map":
        if options == "Single Image":
            st.title("Black & White Depth Estimation - Single Image")
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                output_path = process_single_image_bw(uploaded_file, midas, transform)

                # Show the depth map
                st.image(output_path, caption="Black & White Depth Map", use_column_width=True)

                # Provide a download button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Black & White Depth Map",
                        data=file,
                        file_name="bw_depth_map.png",
                        mime="image/png"
                    )

        elif options == "ZIP File (Dataset)":
            st.title("Black & White Depth Estimation - ZIP File (Dataset)")
            uploaded_zip = st.file_uploader("Upload a ZIP File", type=["zip"])

            if uploaded_zip:
                output_zip_path = process_zip_file_bw(uploaded_zip, midas, transform)

                # Provide a download button
                with open(output_zip_path, "rb") as file:
                    st.download_button(
                        label="Download Processed ZIP",
                        data=file,
                        file_name="processed_bw_dataset.zip",
                        mime="application/zip"
                    )

        elif options == "Video":
            st.title("Black & White Depth Estimation - Video")
            uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

            if uploaded_video:
                st.video(uploaded_video)
                output_video_path = process_video_bw(uploaded_video, midas, transform)

                # Provide a download button
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="processed_video_bw.mp4",
                        mime="video/mp4"
                    )
