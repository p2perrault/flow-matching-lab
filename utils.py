import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from ipywidgets import interact, IntSlider
import os
import cv2
import imageio
from PIL import Image

def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"

def display_image_cli(img_dir, window_name="Slider"):
    # Load and sort images
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not img_files:
        raise ValueError(f"No images found in {img_dir}")

    N = len(img_files)

    # Callback for trackbar
    def on_trackbar(val):
        img = cv2.imread(img_files[val])
        cv2.imshow(window_name, img)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 600)  # optional

    # Create trackbar
    cv2.createTrackbar("step", window_name, 0, N-1, on_trackbar)

    # Show first image
    on_trackbar(0)

    print("Use the slider to move between frames. Press 'q' to quit.")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):  # quit
            break

    cv2.destroyAllWindows()

def display_image_notebook(img_dir):
    # Load and sort image file paths
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not img_files:
        raise ValueError(f"No images found in {img_dir}")

    # Load images into memory
    images = [Image.open(f) for f in img_files]

    # Slider callback
    def show_frame(idx):
        display(images[idx])

    slider = IntSlider(value=0, min=0, max=len(images)-1, step=1, description="Frame")
    interact(show_frame, idx=slider)

def create_bouncing_gif(img_dir, save_gif="animation.gif", fps=30, pause_time=1.5):
    # Load and sort images
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    if not img_files:
        raise ValueError(f"No PNG images found in {img_dir}")

    # Compute pause frames
    pause_frames = int(fps * pause_time)
    N = len(img_files)

    # Bouncing sequence
    forward = list(range(N))
    pause_end = [N-1] * pause_frames
    backward = list(range(N-2, -1, -1))
    pause_start = [0] * pause_frames
    frame_indices = forward + pause_end + backward + pause_start

    # Load all images into memory (BGR -> RGB)
    frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in img_files]

    # Build GIF frames
    gif_frames = [frames[i] for i in frame_indices]

    # Save GIF using imageio
    imageio.mimsave(save_gif, gif_frames, fps=fps, loop=0)
    print(f"GIF saved to {save_gif}")

def live_plot_notebook(data_dict, figsize=(7,5), title=''):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='center left') # the plot evolves to the right
    plt.show()

def live_plot_cli(data_dict, figsize=(7,5), title=''):
    clear_output(wait=True)
    if not hasattr(live_plot_cli, "fig"):
        live_plot_cli.fig, live_plot_cli.ax = plt.subplots(figsize=figsize)
        live_plot_cli.lines = {}

        for label in data_dict:
            line, = live_plot_cli.ax.plot([], [], label=label)
            live_plot_cli.lines[label] = line

        live_plot_cli.ax.legend(loc="center left")
        live_plot_cli.ax.grid(True)
        live_plot_cli.ax.set_xlabel("epoch")

    for label, data in data_dict.items():
        live_plot_cli.lines[label].set_data(range(len(data)), data)

    live_plot_cli.ax.relim()
    live_plot_cli.ax.autoscale_view()
    live_plot_cli.ax.set_title(title)

    plt.pause(.1)


