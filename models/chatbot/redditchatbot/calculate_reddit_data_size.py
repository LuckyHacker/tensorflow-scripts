import requests

url = "http://files.pushshift.io/reddit/comments/"

resp = requests.get(url)

files = resp.text.split("<tr class='file'>")
files = list(map(lambda x: x.split("\n")[3], files))
files = list(map(lambda x: x.split("</a>")[0], files))
files = list(map(lambda x: x if "bz2" in x else None, files))
files = filter(None, files)
files = list(map(lambda x: x.split(">")[-1], files))
files = list(map(lambda x: int(x.replace(",", "")), files))
print(sum(files) / 2**30)
