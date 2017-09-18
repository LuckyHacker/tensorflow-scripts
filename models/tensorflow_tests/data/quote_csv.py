import csv

with open("ETHUSD.csv", "r") as readfile:
    reader = csv.reader(readfile)
    rows = list(reader)
    with open("ETHUSDnew.csv", "w") as writefile:
        writer = csv.writer(writefile, quoting=csv.QUOTE_ALL)
        for row in rows:
            writer.writerow(row)
