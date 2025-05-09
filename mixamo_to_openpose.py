import numpy as np
import cv2
from PIL import Image
import math
import xml.etree.ElementTree as ET
import os
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "Path to folder contianing Mixamo .dae files, or a single .dae file.")
ap.add_argument("-o", "--output", required = True, help = "Path to save outputs.")
ap.add_argument("-ow", "--width", required = False, type=int, default=512, help = "Output image width.")
ap.add_argument("-oh", "--height", required = False, type=int, default=512, help = "Output image height.")
ap.add_argument("-os", "--scale", required = False, type=float, default=2.0, help = "Pose scale multiplier.")
ap.add_argument("-rx", "--rotation_x", required = False, type=int, default=0, help = "Pose X rotation in degrees.")
ap.add_argument("-ry", "--rotation_y", required = False, type=int, default=0, help = "Pose Y rotation in degrees.")
ap.add_argument("-rz", "--rotation_z", required = False, type=int, default=0, help = "Pose Z rotation in degrees.")
ap.add_argument("-ifps", "--input_fps", required = False, type=int, default=30, help = "FPS of Mixamo animation.")
ap.add_argument("-f", "--max_frames", required = False, type=int, default=0, help = "Maximum number of frames in final sequence. Set to 0 for no limit.")
ap.add_argument("-of", "--output_format", required = False, default="GIF", help = "Output format: GIF, PNG, SHEET")
args = vars(ap.parse_args())


# OpenPose keypoints
openpose_keypoints = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear"
]


# Mixamo DAE to OpenPose mappings
dae_to_openpose_map = {
    "mixamorig_HeadTop_End": "nose",
    "mixamorig_Head": "head",  # Not used by OpenPose, exclusively for storing head position and angle for generating nose/eye/ear points
    "mixamorig_Neck": "neck",
    "mixamorig_RightArm": "right_shoulder",
    "mixamorig_RightForeArm": "right_elbow",
    "mixamorig_RightHand": "right_wrist",
    "mixamorig_LeftArm": "left_shoulder",
    "mixamorig_LeftForeArm": "left_elbow",
    "mixamorig_LeftHand": "left_wrist",
    "mixamorig_RightUpLeg": "right_hip",
    "mixamorig_RightLeg": "right_knee",
    "mixamorig_RightFoot": "right_ankle",
    "mixamorig_LeftUpLeg": "left_hip",
    "mixamorig_LeftLeg": "left_knee",
    "mixamorig_LeftFoot": "left_ankle",
    "mixamorig_RightEye": "right_eye",
    "mixamorig_LeftEye": "left_eye",
    "mixamorig_RightEar": "right_ear",
    "mixamorig_LeftEar": "left_ear"
}


# Big boy function for stepping through the DAE file and applying animation transforms
def parse_dae_for_visual_scene(dae_file_path, 
                               nose_offset=(0, 0.2, 0), 
                               eye_offset=(0.2, 0, -0.2), 
                               ear_offset=(0.4, 0.02, -0.45),
                               image_size=(512, 512),
                               rotation_angles=(0, 0, 0),
                               scale_factor=2):
    tree = ET.parse(dae_file_path)
    root = tree.getroot()
    namespace = {'collada': 'http://www.collada.org/2005/11/COLLADASchema'}
    
    visual_scene = root.find('.//collada:library_visual_scenes/collada:visual_scene', namespace)
    animations = root.find('.//collada:library_animations', namespace)
    if visual_scene is None or animations is None:
        raise ValueError("Visual scene or animation data not found in DAE file.")

    # Dictionary to store animation data
    animation_data = {}
    
    # Parse the animation transforms
    for animation in animations.findall('collada:animation', namespace):
        target_id = animation.get('id').split('-')[0]
        times = animation.find(f'.//collada:source[@id="{target_id}-Matrix-animation-input"]/collada:float_array', namespace)
        transforms = animation.find(f'.//collada:source[@id="{target_id}-Matrix-animation-output-transform"]/collada:float_array', namespace)
        
        if times is None or transforms is None:
            continue

        time_values = list(map(float, times.text.split()))
        transform_values = list(map(float, transforms.text.split()))
        matrices = [np.array(transform_values[i:i+16]).reshape(4, 4) for i in range(0, len(transform_values), 16)]
        
        animation_data[target_id] = (time_values, matrices)

    frames = {i: {} for i in range(len(next(iter(animation_data.values()))[0]))}

    # Recursively parse nodes in the visual scene and apply animations
    def parse_joint(node, parent_transform, time_idx):
        joint_name = node.get('id')
        openpose_name = dae_to_openpose_map.get(joint_name)

        # Use animation transform if available, else fall back to the node's local transform
        if joint_name in animation_data:
            _, matrices = animation_data[joint_name]
            local_transform = matrices[time_idx]
        else:
            matrix_text = node.find('collada:matrix', namespace).text
            matrix_values = list(map(float, matrix_text.split()))
            local_transform = np.array(matrix_values).reshape(4, 4)
        
        # Compute the world transform for the joint at this time step
        world_transform = np.dot(parent_transform, local_transform)
        x, y, z = world_transform[0, 3], -world_transform[1, 3], world_transform[2, 3]

        if openpose_name:
            frames[time_idx][openpose_name] = [x, y, z]
            if openpose_name == "nose":  # The head's orientation is based on the nose position
                frames[time_idx]["head_rotation_matrix"] = world_transform[:3, :3]

        for child in node.findall('collada:node', namespace):
            parse_joint(child, world_transform, time_idx)

    # Parse the visual scene for each time step
    for time_idx in range(len(next(iter(animation_data.values()))[0])):
        root_node = visual_scene.find('.//collada:node[@id="mixamorig_Hips"]', namespace)
        parse_joint(root_node, np.eye(4), time_idx)

    # Add facial feature points based on the head orientation and position
    for time_idx, frame in frames.items():
        head = frame.get("nose")
        neck = frame.get("neck")
        head_rotation_matrix = frame.get("head_rotation_matrix", np.eye(3))  # Default to identity if not found

        if head and neck:
            neck_to_nose_dist = np.linalg.norm(np.array(head) - np.array(neck))
            nose_offset_scaled = np.array(nose_offset) * neck_to_nose_dist
            eye_offset_scaled = np.array(eye_offset) * neck_to_nose_dist
            ear_offset_scaled = np.array(ear_offset) * neck_to_nose_dist

            # Apply inverse rotation to adjust direction properly
            adjusted_rotation_matrix = np.linalg.inv(head_rotation_matrix)
            adjusted_rotation_matrix[0, 2] *= -1  # Invert yaw on the Z-axis for left-right direction
            adjusted_rotation_matrix[2, 0] *= -1

            def rotate_relative_to_head(offset):
                return adjusted_rotation_matrix @ offset

            rotated_nose = rotate_relative_to_head(nose_offset_scaled)
            rotated_right_eye = rotate_relative_to_head(eye_offset_scaled)
            rotated_left_eye = rotate_relative_to_head([-eye_offset_scaled[0], eye_offset_scaled[1], eye_offset_scaled[2]])
            rotated_right_ear = rotate_relative_to_head(ear_offset_scaled)
            rotated_left_ear = rotate_relative_to_head([-ear_offset_scaled[0], ear_offset_scaled[1], ear_offset_scaled[2]])

            frame["nose"] = [head[0] + rotated_nose[0], head[1] + rotated_nose[1], head[2] + rotated_nose[2]]
            frame["right_eye"] = [head[0] + rotated_right_eye[0], head[1] + rotated_right_eye[1], head[2] + rotated_right_eye[2]]
            frame["left_eye"] = [head[0] + rotated_left_eye[0], head[1] + rotated_left_eye[1], head[2] + rotated_left_eye[2]]
            frame["right_ear"] = [head[0] + rotated_right_ear[0], head[1] + rotated_right_ear[1], head[2] + rotated_right_ear[2]]
            frame["left_ear"] = [head[0] + rotated_left_ear[0], head[1] + rotated_left_ear[1], head[2] + rotated_left_ear[2]]

    # Rotate and scale all points in the frames
    frames = rotate_and_scale_pose(frames, image_size, rotation_angles, scale_factor)

    # Sort frames
    sorted_frames = [frames[frame] for frame in sorted(frames.keys())]
    return sorted_frames


# Apply rotations to all keypoints around the image center
def rotate_and_scale_pose(frames, image_size, rotation_angles, scale_factor=1.0):
    # Convert angles to radians
    rx, ry, rz = np.radians(rotation_angles)
    
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix in the order: Ry -> Rx -> Rz
    R = Rz @ (Rx @ Ry)

    # Apply rotation and scaling to each point in each frame
    for time_idx, frame in frames.items():
        for point_key, coords in frame.items():
            # Ensure coords is a list of three elements before proceeding
            if isinstance(coords, list) and len(coords) == 3:
                x, y, z = coords
                
                # Apply scaling
                x, y, z = x * scale_factor, y * scale_factor, z * scale_factor
                
                # Rotate the point around the origin (0, 0, 0)
                rotated_point = R @ np.array([x, y, z])
                
                # Update the point with rotated and scaled coordinates
                frame[point_key] = rotated_point.tolist()

    # Center the points on the specified canvas size
    for time_idx, frame in frames.items():
        for point_key, coords in frame.items():
            # Shift x and y only if coords is a list of three elements
            if isinstance(coords, list) and len(coords) == 3:
                frame[point_key][0] += image_size[0] // 2  # Shift x to center
                frame[point_key][1] += image_size[1] // 2  # Shift y to center

    return frames


# Center keypoints on image canvas
def center_keypoints(frames, canvas_size=(512, 512)):
    canvas_center_x, canvas_center_y = canvas_size[0] // 2, canvas_size[1] // 2
    
    for frame in frames:
        # Calculate the average position of keypoints for centering
        all_points = np.array([frame[keypoint][:2] for keypoint in frame if keypoint in openpose_keypoints])
        avg_x, avg_y = np.mean(all_points, axis=0)

        # Adjust each keypoint to center the entire pose
        for keypoint in frame:
            frame[keypoint][0] += (canvas_center_x - avg_x)
            frame[keypoint][1] += (canvas_center_y - avg_y)
    
    return frames


# Format frame keypoints to OpenPose JSON standard
def format_to_openpose(frames):
    formatted_frames = []
    for frame in frames:
        candidate = []
        subset = []
        
        for keypoint in openpose_keypoints:
            if keypoint in frame:
                x, y, _ = frame[keypoint]
                candidate.append([x, y, 1.0])
            else:
                candidate.append([0.0, 0.0, 0.0])

        subset.append([i if candidate[i][2] > 0 else -1 for i in range(len(openpose_keypoints))])
        formatted_frames.append({"candidate": candidate, "subset": subset})
    
    return formatted_frames


# Convert DAE to OpenPose frames with rotation and scaling
def convert_dae_to_openpose(dae_file, image_size=(512, 512), rotation_angles=(0, 0, 0), scale_factor=2):
    frames = parse_dae_for_visual_scene(dae_file, image_size=image_size, rotation_angles=rotation_angles, scale_factor=scale_factor)
    centered_frames = center_keypoints(frames, canvas_size=image_size)
    openpose_frames = format_to_openpose(centered_frames)
    return openpose_frames


# Load keypoints from the OpenPose JSON standard
def load_keypoints_for_drawing(data):
    # Load keypoints for each frame to prepare for drawing
    frames = []
    for frame in data:
        candidate = np.array(frame['candidate'])
        subset = np.array(frame['subset'])
        frames.append((candidate, subset))
    
    return frames


# Draw the body keypoint and limbs
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 2
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    # Draw keypoints with true sub-pixel positions
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            # Use exact floating-point coordinates without rounding
            cv2.circle(canvas, (int(x), int(y)), 2, colors[i], thickness=-1, lineType=cv2.LINE_AA)
    
    # Draw limbs with true sub-pixel positions
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            # Use exact floating-point coordinates without rounding
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i], lineType=cv2.LINE_AA)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


# Reduce the number of frames to a maximum with even distribution
def reduce_frames(frames, max_frames):
    """
    Reduces the number of frames to `max_frames` by keeping frames with as even a distribution as possible.
    """
    num_frames = len(frames)
    if num_frames <= max_frames:
        return frames  # No reduction needed

    # Calculate approximate step size and round indices
    step = num_frames / max_frames
    reduced_frames = [frames[round(i * step)] for i in range(max_frames)]
    
    return reduced_frames


#Convert dae file and save as image sequence
def convert_dae(input_path, output_path, image_size=(320, 512), rotation_angles=(0, 0, 0), scale_factor=2, input_fps=30, max_frames=0, output_format='GIF'):
    # Original frame duration based on FPS
    original_frame_duration = int(1000 / input_fps)

    # Determine if input is a single file or a folder
    if os.path.isfile(input_path) and input_path.lower().endswith('.dae'):
        dae_files = [input_path]
        # Set default output folder if only a file name is provided
        output_folder = os.path.dirname(output_path) or os.getcwd()
        os.makedirs(output_folder, exist_ok=True)
        output_paths = [os.path.join(output_folder, os.path.basename(output_path))]
    elif os.path.isdir(input_path):
        dae_files = glob.glob(os.path.join(input_path, "*.dae"))
        os.makedirs(output_path, exist_ok=True)
        output_paths = [os.path.join(output_path, os.path.splitext(os.path.basename(dae_file))[0] + f".{output_format.lower()}") for dae_file in dae_files]
    else:
        raise ValueError("Input path must be a .dae file or a directory containing .dae files.")

    # Process each .dae file
    for dae_file, output_file_path in zip(dae_files, output_paths):
        print(f"Converting {dae_file} to OpenPose")
        frames = load_keypoints_for_drawing(convert_dae_to_openpose(dae_file, image_size=image_size, rotation_angles=rotation_angles, scale_factor=scale_factor))
        
        # Reduce frames if necessary
        original_num_frames = len(frames)
        if max_frames > 0:
            frames = reduce_frames(frames, max_frames)
        else:
            max_frames = len(frames)
        
        # Calculate adjusted frame duration for GIF
        if output_format == 'GIF' and len(frames) > 1:
            frame_duration = int(original_frame_duration * (original_num_frames / len(frames)))
        else:
            frame_duration = original_frame_duration

        pil_images = []
        for idx, (candidate, subset) in enumerate(frames):
            canvas = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
            drawn_canvas = draw_bodypose(canvas, candidate, subset)
            drawn_canvas_rgb = cv2.cvtColor(drawn_canvas, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(drawn_canvas_rgb)
            pil_images.append(pil_image)
            
            # Save each frame as a PNG if output format is PNG
            if output_format == 'PNG':
                frame_folder = os.path.join(os.path.splitext(output_file_path)[0])
                os.makedirs(frame_folder, exist_ok=True)
                frame_path = os.path.join(frame_folder, f"{idx:0{len(str(max_frames))}d}.png")
                pil_image.save(frame_path)
        
        # Save as GIF
        if output_format == 'GIF':
            pil_images[0].save(
                output_file_path, 
                save_all=True, 
                append_images=pil_images[1:], 
                duration=frame_duration, 
                loop=0
            )
            print(f"GIF saved at {output_file_path} with a maximum of {max_frames} frames")
        
        # Save as image sheet (SHEET)
        elif output_format == 'SHEET':
            output_file_path = f"{os.path.splitext(output_file_path)[0]}.png"
            grid_size = int(np.ceil(np.sqrt(len(pil_images))))
            sheet_width = grid_size * image_size[0]
            sheet_height = grid_size * image_size[1]
            sheet_image = Image.new('RGB', (sheet_width, sheet_height), (0, 0, 0))

            for idx, pil_image in enumerate(pil_images):
                row = idx // grid_size
                col = idx % grid_size
                sheet_image.paste(pil_image, (col * image_size[0], row * image_size[1]))

            sheet_image.save(output_file_path)
            print(f"Image sheet saved at {output_file_path}")


if os.path.isfile(args["input"]):
    convert_dae(args["input"], args["output"], image_size=(args["width"], args["height"]), rotation_angles=(args["rotation_x"], args["rotation_y"], args["rotation_z"]), scale_factor=args["scale"], input_fps=args["input_fps"], max_frames=args["max_frames"], output_format=args["output_format"])
else:
    print(f"Input \"{args['input']}\" is not a file")