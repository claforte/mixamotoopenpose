# Mixamo to OpenPose
Convert Mixamo animations directly to OpenPose image sequences

Mixamo (https://www.mixamo.com/) is a massive library of ready-made human skeleton animations, commonly used in VFX and 3D games.
With this script you can easily convert Mixamo .dae (Collada) files into sequences of images (gifs, single image sheets) using the OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose) bones system.

![Example1](https://github.com/user-attachments/assets/d2f0ef5a-aca7-4566-9542-2a4861eeb22e)

Animations can be reduced to a specific number of frames, rotated to be viewed from any angle, and scaled to any size.

This script allows the easy creation of OpenPose animations for AI image generation, or dataset creation.

# Usage

## Easy Mode
Follow the "instructions.txt" inside the "EasyMode" folder to easily install and run the conversion script.

## CLI

Requirements: Pillow, Numpy, OpenCV

`pip install pillow numpy opencv-python`

Download an animation from Mixamo as a Collada .dae file. (Recommended without skin for smaller file size.)
Extract the zip folder so you have the standard .dae file itself.

Run mixamo_to_openpose.py with the required input and output arguments

Arguments:

- -i --input: Expects string, Path to folder contianing Mixamo .dae files, or a single .dae file.

- -o --output: Expects string, Path to save outputs.

- -ow --width: Expects int, Output image width. Defaults to 512.

- -oh --height: Expects int, Output image height. Defaults to 512.

- -os --scale: Expects float, Pose scale multiplier. Adjusts the scale of the pose in the output. Defaults to 2.0.

- -rx --rotation_x: Expects int, Pose X-axis rotation in degrees. Controls the pose's rotation along the X-axis. Defaults to 0.

- -ry --rotation_y: Expects int, Pose Y-axis rotation in degrees. Controls the pose's rotation along the Y-axis. Defaults to 0.

- -rz --rotation_z: Expects int, Pose Z-axis rotation in degrees. Controls the pose's rotation along the Z-axis. Defaults to 0.

- -ifps --input_fps: Expects int, FPS of Mixamo animation. Specifies the frame rate for the input animation, in frames per second. Defaults to 30.

- -f --max_frames: Expects int, Maximum number of frames in the final sequence. Limits the total frames in the output; set to 0 for no limit. Defaults to 0.

- -of --output_format: Expects string, Output format for saved images. Options are "GIF" (single animated GIF), "PNG" (folder with numbered PNG images for each frame), or "SHEET" (one image sheet containing all frames arranged in a grid). Defaults to "GIF."

# More examples
![Example2](https://github.com/user-attachments/assets/ed94e49e-fcee-49ad-82e9-b8588d84cdf9)

![walk](https://github.com/user-attachments/assets/103ff122-0485-4f11-90cc-033626ff6633)



