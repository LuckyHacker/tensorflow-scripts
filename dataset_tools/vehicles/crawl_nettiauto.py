# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from collections import OrderedDict
import requests
import csv
import time


total_pages = 3017
output_file = "output.csv"
search_classes = [  "make_model_link",
                    "main_price",
                    "checkLnesFlat",
                    "vehicle_other_info clearfix_nett",

                    ]

csv_column_names = ["model", "desc", "engine", "year", "kilometers", "fuel", "transmission", "traffic_legal", "price"]
url = "https://www.nettiauto.com/vaihtoautot?page={}"

with open(output_file, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(csv_column_names)

    for page_num in range(1, total_pages + 1):
        page_load_begin = time.time()
        dataset = OrderedDict(  [
                                ("model", ([])),
                                ("desc", ([])),
                                ("engine", ([])),
                                ("year", ([])),
                                ("kilometers", ([])),
                                ("fuel", ([])),
                                ("transmission", ([])),
                                ("traffic_legal", ([])),
                                ("price", ([])),
                                ]
                                )

        r = requests.get(url.format(page_num))
        page_load_end = round(time.time() - page_load_begin, 2)

        page_parse_begin = time.time()
        soup = BeautifulSoup(r.text, 'html.parser')
        data_boxes = soup.find_all('div', { "class" : "data_box" })
        for data_box in data_boxes:
            for cl in search_classes:
                data = data_box.find('div', { "class" : cl })

                if cl == "make_model_link":

                    model = data.find(text=True, recursive=False)
                    if model:
                        dataset["model"].append(model)
                    else:
                        dataset["model"].append("")

                    engine = data.find('span', { "class" : "eng_size"}).get_text()
                    if engine:
                        engine = engine.replace("(", "")
                        engine = engine.replace(")", "")
                        engine = engine.replace(" ", "")
                        dataset["engine"].append(engine)
                    else:
                        dataset["engine"].append("")

                if cl == "main_price":
                    price = data.get_text()
                    price = price.replace(" ", "")
                    dataset["price"].append(price)

                if cl == "checkLnesFlat":
                    try:
                        desc = data.find_all(text=True, recursive=False)[1]
                        desc = desc.replace("\n", "")
                    except:
                        desc = ""

                    dataset["desc"].append(desc)

                    traffic_legal = data.find('span')
                    if traffic_legal:
                        traffic_legal = "False"
                    else:
                        traffic_legal = "True"
                    dataset["traffic_legal"].append(traffic_legal)

                if cl == "vehicle_other_info clearfix_nett":
                    other_info = data.find_all('li')

                    if len(other_info) == 4:
                        dataset["year"].append(other_info[0].get_text())
                        dataset["kilometers"].append(other_info[1].get_text())
                        dataset["fuel"].append(other_info[2].get_text())
                        dataset["transmission"].append(other_info[3].get_text())

                    elif len(other_info) == 3:
                        dataset["year"].append(other_info[0].get_text())
                        dataset["fuel"].append(other_info[1].get_text())
                        dataset["transmission"].append(other_info[2].get_text())

        writer.writerows(zip(*dataset.values()))
        page_parse_end = round(time.time() - page_parse_begin, 2)
        print("Page {}, out of {}: Loaded in {}s, parsed in {}s".format(page_num,
                                                                        total_pages,
                                                                        page_load_end,
                                                                        page_parse_end))
