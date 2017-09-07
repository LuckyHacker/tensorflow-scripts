from PIL import Image
from fractions import gcd

with open("food-101/meta/train.txt", "r") as f:
    paths = f.read().split("\n")[:-1]

paths = list(map(lambda x: "food-101/images/{}.jpg".format(x), paths))

sizes = []
ar = []

for path in paths:
    sizes.append(Image.open(path).size)

for x, y in sizes:
    g = gcd(x, y)
    ar.append("{}:{}".format(int(x / g), int(y / g)))

for i in range(50):
    most_frequent = max(set(ar), key=ar.count)
    print("{}. used size: {}".format(i+1, most_frequent))
    ar = list(filter((most_frequent).__ne__, ar))
