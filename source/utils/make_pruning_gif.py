import imageio
import os
import re

# specify the directory where the images are
from source.utils.Configuration import Configuration

directory = os.path.join(Configuration.RES_DIR, "adult/black_box_defense/plots/attacked_models/3hl_5s/0.5ratio_1rep")

# get a list of all the png files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.png')]

# sort the files by the number in their name (assuming the format is 'number%.png')
files.sort(key=lambda f: int(re.search(r'(\d+)%', f).group(1)))

# create a list of images
images = [imageio.imread(os.path.join(directory, f)) for f in files]

# write the images to a gif
imageio.mimsave(os.path.join(directory, 'output.gif'), images)