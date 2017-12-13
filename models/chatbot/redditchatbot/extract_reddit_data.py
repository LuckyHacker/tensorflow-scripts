import bz2
from datetime import datetime
from dateutil.relativedelta import relativedelta

startDate = '2014-01'
endDate = '2014-03'
dir_path = "/media/onni/2tb/"


def generate_dates():
    print("Generating all dates from {} to {}".format(startDate, endDate))
    start = datetime.strptime(startDate, '%Y-%m').date()
    end = datetime.strptime(endDate, '%Y-%m').date()
    all_dates = [startDate]
    cur_date = start

    while cur_date < end:
        cur_date += relativedelta(months=1)
        all_dates.append("-".join(str(cur_date).split("-")[:-1]))

    return all_dates


def generate_filepaths(all_dates):
    print("Generating filepaths")
    all_filepaths = []
    for date in all_dates:
        all_filepaths.append((  "{}RC_{}.bz2".format(dir_path, date),
                                "{}RC_{}".format(dir_path, date)))

    return all_filepaths


def extract_all_files(all_filepaths):
    print("Extracting all files")
    for from_path, to_path in all_filepaths:
        print("Extracting file {} to {}".format(from_path, to_path))
        with open(from_path, "rb") as from_file:
            with open(to_path, "wb") as to_file:
                decompressor = bz2.BZ2Decompressor()
                for data in iter(lambda : from_file.read(100 * 1024), b''):
                    to_file.write(decompressor.decompress(data))



if __name__ == "__main__":
    all_dates = generate_dates()
    all_filepaths = generate_filepaths(all_dates)
    extract_all_files(all_filepaths)
