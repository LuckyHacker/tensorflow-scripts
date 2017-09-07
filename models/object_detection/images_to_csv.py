import os
from PIL import Image


def image_to_csv(path, target_file):
    filename = "/".join(path.split("/")[-2:])
    width, height = str(Image.open(path).size[0]), str(Image.open(path).size[1])
    c = str(path.split("/")[-2])
    xmin = str(0)
    ymin = str(0)
    xmax = str(width)
    ymax = str(height)

    attr = [filename, width, height, c, xmin, ymin, xmax, ymax]

    with open(target_file, "a") as f:
        f.write(",".join(attr) + "\n")


def main():
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    target_file = 'data/{}_labels.csv'.format("lul")
    images_path = "food-101/images/"
    paths_file = "food-101/meta/train.txt"

    with open(target_file, "w") as f:
        f.write(",".join(column_name) + "\n")

    with open(paths_file, "r") as f:
        paths = f.read().split("\n")[:-1]

    image_paths = list(map(lambda x: "{}.jpg".format(images_path + x), paths))
    for image_path in image_paths:
        image_path = os.path.join(os.getcwd(), image_path)
        print(image_path)
        image_to_csv(image_path, target_file)
    print('Successfully exported images to csv.')


main()
