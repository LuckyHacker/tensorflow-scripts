# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from collections import OrderedDict
import requests
import csv
import time


total_pages = 5858
output_file = "output.csv"
search_classes = ["address", "type", "size", "price", "year"]
csv_column_names = ["city", "address", "type", "size", "price", "year"]
url = "https://www.etuovi.com/myytavat-asunnot/tulokset?haku=M1142921947&page={}"

with open(output_file, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(csv_column_names)
    for page_num in range(1, total_pages + 1):
        page_load_begin = time.time()
        dataset = OrderedDict(  [
                                ("city", ([])),
                                ("address", ([])),
                                ("type", ([])),
                                ("size", ([])),
                                ("price", ([])),
                                ("year", ([])),
                                ]
                                )


        r = requests.get(url.format(page_num))
        page_load_end = round(time.time() - page_load_begin, 2)

        page_parse_begin = time.time()
        soup = BeautifulSoup(r.text, 'html.parser')
        houses = soup.find_all('a', { "class" : "facts" })

        for house in houses:
            divs = house.find_all('div')
            for div in divs:
                for cl in search_classes:
                    attrs = div.find_all("div", { "class" : cl})
                    for attr in attrs:

                        if cl == "address":
                            dataset["address"].append(attr.find("strong").get_text())
                            dataset["city"].append(attr.find("span").get_text())

                        if cl == "type":
                            dataset["type"].append(attr.find("label").get_text())

                        if cl == "size":
                            dataset["size"].append(attr.find("span").get_text())

                        if cl == "price":
                            dataset["price"].append(attr.find("span").get_text())

                        if cl == "year":
                            try:
                                dataset["year"].append(attr.find("span").get_text())
                            except:
                                dataset["year"].append("")

        writer.writerows(zip(*dataset.values()))
        page_parse_end = round(time.time() - page_parse_begin, 2)
        print("Page {}, out of {}: Loaded in {}s, parsed in {}s".format(page_num,
                                                                        total_pages,
                                                                        page_load_end,
                                                                        page_parse_end))
