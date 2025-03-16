"""Record3D visualizer

Parse and stream record3d captures. To get the demo data, see `./assets/download_record3d_dance.sh`.
"""

import argparse
import pickle
import time
import cv2
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import viser
import viser.transforms as tf

def downsample(position, color, flys, downsample_factor):
    new_height = position.shape[0] // downsample_factor
    new_width = position.shape[1] // downsample_factor
    position_downsampled = cv2.resize(position, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    color_downsampled = cv2.resize(color, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    flys_downsampled = cv2.resize(flys, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    flys_downsampled = np.round(flys_downsampled).astype(np.float32)

    return position_downsampled, color_downsampled, flys_downsampled


def main(
    xyzs, 
    rgbs, 
    flys, 
    intrinsics, 
    extrinsics,
    default_fps: int=30,
    downsample_factor: int = 1,
    share: bool = False,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading frames!")
    num_frames = xyzs.shape[0]

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.001,
            max=0.02,
            step=1e-3,
            initial_value=0.01,
        )
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=default_fps
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            # Toggle visibility.
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!


    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([-np.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    point_nodes: list[viser.PointCloudHandle] = []
    for i in tqdm(range(num_frames)):
        position = xyzs[i]       # (H, W, 3)
        color = rgbs[i]          # (H, W, 3)
        fly = flys[i]           # (H, W)

        position, color, fly = downsample(position, color, fly, downsample_factor)

        # position_valid = position[fly > 0].reshape(-1, 3)
        # color_valid = color[fly > 0].reshape(-1, 3)
        position_valid = position.reshape(-1, 3)
        color_valid = color.reshape(-1, 3)

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/point_cloud",
                points=position_valid,
                colors=color_valid,
                point_size=gui_point_size.value,
                point_shape="rounded",
            )
        )

        # Place the frustum.
        fov = 2 * np.arctan2(rgbs[i].shape[0] / 2, intrinsics[i, 0, 0] * rgbs[i].shape[1])
        aspect = rgbs[i].shape[1] / rgbs[i].shape[0]
        camera_handle = server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.25,
            image=rgbs[i][::downsample_factor, ::downsample_factor],
            wxyz=tf.SO3.from_matrix(extrinsics[i][:3, :3]).wxyz,
            position=extrinsics[i][:3, 3],
        )
        camera_handle.controls_enabled = True

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.05,
            axes_radius=0.005,
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        # Update the timestep if we're playing.
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        point_nodes[gui_timestep.value].point_size = gui_point_size.value
        point_nodes[
            (gui_timestep.value + 1) % num_frames
        ].point_size = gui_point_size.value

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Viser Visualization")
    parser.add_argument("-f", "--pickle_file", type=str)
    args = parser.parse_args()

    with open(args.pickle_file, 'rb') as f:
        result = pickle.load(f)
    xyzs, rgbs, flys, intrinsics, extrinsics = result['xyzs'], result['rgbs'], result['flys'], result['intrinsics'], result['extrinsics']
    
    main(xyzs, rgbs, flys, intrinsics, extrinsics)

# python -m egomono4d.demo_viser -f result_example_epic_kitchen.pkl
