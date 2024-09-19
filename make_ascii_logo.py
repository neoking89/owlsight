# # import sys
# # import numpy as np
# # from PIL import Image

# # # Contrast on a scale -10 -> 10
# # contrast = 10
# # density = ('$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|'
# #            '()1{}[]?-_+~<>i!lI;:,"^`\'.            ')
# # density = density[:-11+contrast]
# # n = len(density)

# # img_name = sys.argv[1]
# # try:
# #     width = int(sys.argv[2])
# # except IndexError:
# #     # Default ASCII image width.
# #     width = 100

# # # Read in the image, convert to greyscale.
# # img = Image.open(img_name)
# # img = img.convert('L')
# # # Resize the image as required.
# # orig_width, orig_height = img.size
# # r = orig_height / orig_width
# # # The ASCII character glyphs are taller than they are wide. Maintain the aspect
# # # ratio by reducing the image height.
# # height = int(width * r * 0.5)
# # img = img.resize((width, height), Image.Resampling.LANCZOS)

# # # Now map the pixel brightness to the ASCII density glyphs.
# # arr = np.array(img)
# # for i in range(height):
# #     for j in range(width):
# #         p = arr[i,j]
# #         k = int(np.floor(p/256 * n))
# #         print(density[n-1-k], end='')
# #     print()

# import sys
# import numpy as np
# from PIL import Image

# # Contrast on a scale -10 -> 10
# contrast = 10
# density = ('$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|'
#            '()1{}[]?-_+~<>i!lI;:,"^`\'.            ')
# density = density[:-11 + contrast]
# n = len(density)

# img_name = sys.argv[1]

# # Control scale with a scaling factor
# try:
#     width = int(sys.argv[2])
# except IndexError:
#     # Default ASCII image width.
#     width = 100

# # Optional height scaling factor
# try:
#     scale_factor = float(sys.argv[3])
# except IndexError:
#     # Default scale factor.
#     scale_factor = 0.5

# # Read in the image, convert to greyscale.
# img = Image.open(img_name)
# img = img.convert('L')

# # Resize the image while maintaining aspect ratio.
# orig_width, orig_height = img.size
# aspect_ratio = orig_height / orig_width
# height = int(width * aspect_ratio * scale_factor)
# img = img.resize((width, height), Image.Resampling.LANCZOS)

# # Map the pixel brightness to the ASCII density glyphs.
# arr = np.array(img)

# for i in range(height):
#     for j in range(width):
#         p = arr[i, j]
#         k = int(np.floor(p / 256 * n))
#         print(density[n - 1 - k], end='')
#     print()

import sys
import numpy as np
from PIL import Image

# Contrast on a scale -10 -> 10
contrast = 10
density = (
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|"
    "()1{}[]?-_+~<>i!lI;:,\"^`'.            "
)
density = density[:-11 + contrast]
n = len(density)

img_name = sys.argv[1]

# Control scale with a scaling factor
try:
    width = int(sys.argv[2])
except IndexError:
    # Default ASCII image width.
    width = 100

# Optional height scaling factor
try:
    scale_factor = float(sys.argv[3])
except IndexError:
    # Default scale factor.
    scale_factor = 0.5

# Read in the image, convert to greyscale.
img = Image.open(img_name)
img = img.convert('L')

# Resize the image while maintaining aspect ratio.
orig_width, orig_height = img.size
aspect_ratio = orig_height / orig_width
height = int(width * aspect_ratio * scale_factor)
img = img.resize((width, height), Image.Resampling.LANCZOS)

# Map the pixel brightness to the ASCII density glyphs.
arr = np.array(img)

ascii_art = []
for i in range(height):
    row = []
    for j in range(width):
        p = arr[i, j]
        k = int(np.floor(p / 256 * n))
        row.append(density[n - 1 - k])
    ascii_art.append(''.join(row))

# Print the ASCII art
print('\n'.join(ascii_art))