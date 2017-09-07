import os

label = "meatball"
path = "/home/onni/Downloads/meatball_dataset/"
files = os.listdir(path)

counter = 0
for fi in files:
    os.rename(path + fi, path + label + "{}.".format(counter) + fi.split(".")[-1])
    counter += 1
