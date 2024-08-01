import json

from PIL import Image, ImageDraw

data = json.load(open("random_triangles.json", "r"))

width = 4096
height = 4096

def unpack_color(col):
    return tuple(map(lambda i: max(0, min(255, int(i * 255.99))), col))

# https://github.com/4dcu-be/Genetic-Art-Algorithm/blob/master/painting.py#L49
image = Image.new("RGBA", (width, height))
draw = ImageDraw.Draw(image)

background = unpack_color(data["background"])

draw.polygon([(0, 0), (0, height), (width, height), (width, 0)],
                fill=background)

for t in data["triangles"]:
    new_triangle = Image.new("RGBA", (width, height))
    tdraw = ImageDraw.Draw(new_triangle)
    tdraw.polygon([(x*width, y*height) for x, y in t["vertices"]], fill=unpack_color(t["colour"]))

    image = Image.alpha_composite(image, new_triangle)

image.save("output.png")