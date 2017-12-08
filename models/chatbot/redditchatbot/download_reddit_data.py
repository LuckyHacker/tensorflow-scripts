from datetime import datetime
from dateutil.relativedelta import relativedelta
import requests

startDate = '2005-12'
endDate = '2017-10'


def generate_dates():
    start = datetime.strptime(startDate, '%Y-%m').date()
    end = datetime.strptime(endDate, '%Y-%m').date()
    all_dates = [startDate]
    cur_date = start

    while cur_date < end:
        cur_date += relativedelta(months=1)
        all_dates.append("-".join(str(cur_date).split("-")[:-1]))

    return all_dates


def generate_urls(all_dates):
    all_urls = []
    for date in all_dates:
        all_urls.append("http://files.pushshift.io/reddit/comments/RC_{}.bz2".format(date))

    return all_urls


def download_file(url):
    # https://stackoverflow.com/a/16696317
    local_filename = "/media/onni/e4f9824a-a2e1-43d7-bc30-c772be2fb9f2/" + url.split('/')[-1]
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return local_filename


def download_all(all_urls):
    for url in all_urls:
        download_file(url)

if __name__ == "__main__":
    all_dates = generate_dates()
    all_urls = generate_urls(all_dates)
    download_all(all_urls)
