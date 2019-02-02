import string
from tkinter import *

import PIL
import numpy as np
from PIL import Image, ImageDraw

from constants import POINTS_DIR, IMAGES_DIR, WIDTH, HEIGHT

white = (255, 255, 255)
green = (0, 128, 0)


def draw_image(points):
    image = PIL.Image.new("RGB", (WIDTH, HEIGHT), white)
    draw = ImageDraw.Draw(image)
    for (y, x) in points:
        draw.ellipse([(x - 1, y - 1), (x + 1, y + 1)], fill='black', width=0)
    return image


def save():
    global char_id
    char_name = string.ascii_lowercase[char_id]
    filepath = (IMAGES_DIR / char_name).with_suffix(".jpg")
    image = draw_image(points)
    image.save(filepath)
    np.savetxt((POINTS_DIR / char_name).with_suffix(".txt"), points, fmt='%d', delimiter=',', header='y,x')
    print(f"Saved {filepath}")
    cv.delete('all')
    points.clear()
    char_id += 1
    if char_id >= len(string.ascii_lowercase):
        cv.quit()
        quit()
    cv.create_text(5, 5, text=string.ascii_lowercase[char_id])


def paint(event):
    points.append([event.y, event.x])
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=1)


root = Tk()
cv = Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)
char_id = 0
cv.create_text(5, 5, text=string.ascii_lowercase[char_id])

points = []
button = Button(text="save", command=save)
button.pack()
root.mainloop()
