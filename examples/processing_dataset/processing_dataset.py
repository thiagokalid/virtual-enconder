from src.visual_odometer import VisualOdometer
import numpy as np
import os


def read_configs(filename):
    import json
    with open(filename, 'r') as file:
        # Load the JSON data into a Python dictionary
        data = json.load(file)
    return data


def load_imgs(folder_name):
    import os
    import numpy as np
    from PIL import Image

    image_arrays = []

    image_files = sorted(
        [f for f in os.listdir(folder_name) if f.endswith(".jpg")],
        key=lambda x: int(x.split("_")[0][5:])
    )
    for filename in image_files:
        img = Image.open(os.path.join(folder_name, filename)).convert('L')  # 'L' mode is grayscale
        # Convert the image to a numpy array and append to the list
        img_array = np.array(img)
        image_arrays.append(img_array)

    return image_arrays


specimen_type = ["planar"]
medium_type = ["water", "air"]
sweep_type = ["single_x", "single_y", "closed_loop"]
folder_names = [
    specimen + "/" + medium + "_" + sweep
    for specimen in specimen_type
    for medium in medium_type
    for sweep in sweep_type
]
configs = read_configs("configs.json")

for index, config in zip(configs.keys(), configs.values()):
    for folder_name in folder_names:
        print(f"Processing {folder_name}, Config: {index}.")
        img_stream = load_imgs("data/" + folder_name)
        n_imgs = len(img_stream)

        # Create odometer object:
        odometer = VisualOdometer(img_size=(640, 480))
        odometer.set_config(config)

        displacements = []

        # Process the frames:
        for img in img_stream:
            odometer.feed_image(img)
            displacement = odometer.get_displacement()
            displacements.append(displacement)

        # Check if the directory exists; if not, create it
        config_path = f"output/config{index}"
        output_dir = config_path + "/" + folder_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save locally the disclacements:
        odometer.save_config(path=config_path)
        np.save(output_dir + "/displacements.npy", displacements)
