# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import requests
import csv


total_pages = 5858
output_file = "output.csv"
search_classes = ["address", "type", "size", "price", "year"]
csv_column_names = ["city", "address", "type", "size", "price", "year"]

with open(output_file, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(csv_column_names)
    for page_num in range(1, total_pages + 1):

        dataset = { "city": [],
                    "address": [],
                    "type": [],
                    "size": [],
                    "price": [],
                    "year": [],
                    }

        print("Page: {}".format(page_num))
        url = "https://www.etuovi.com/myytavat-asunnot/tulokset?haku=M1142921947&page={}".format(page_num)
        r = requests.get(url)
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
