

with open("movie_lines.txt", "rb") as f:
    lines = f.readlines()


for line in lines:
    line = str(line, "latin-1")
    utter = line.split("+++$+++")[-1][1:]
    movie_index = line.split("+++$+++")[-3][1:-1]
    with open(movie_index, "a") as f:
        f.write(utter)
