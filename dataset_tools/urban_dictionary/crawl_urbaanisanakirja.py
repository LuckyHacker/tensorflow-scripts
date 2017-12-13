import requests
import string
import json

from bs4 import BeautifulSoup as bs

save_location = "quotes.json"

alphabet = string.ascii_lowercase + "åäö"

base_url = "https://urbaanisanakirja.com"
base_browse_url = "https://urbaanisanakirja.com/browse/{}/?page={}"


def get_number_of_pages(url):
    resp = requests.get(url)
    html_doc = resp.text
    soup = bs(html_doc, 'html.parser')
    li_elems = soup.find_all("li")
    page_li_elems = [li for li in li_elems if "?page=" in str(li)]
    page_li_strings = list(map(lambda x: x.a.string, page_li_elems))
    page_nums = [num for num in page_li_strings if num.isdigit()]

    try:
        max_page = max(page_nums)
    except ValueError:
        max_page = 1

    return max_page


def get_word_links(url):
    resp = requests.get(url)
    html_doc = resp.text
    soup = bs(html_doc, 'html.parser')
    tr_elems = soup.find_all("tr")
    a_elems = list(map(lambda x: x.a, tr_elems))
    word_links = [a["href"] for a in a_elems if "/word" in str(a)]
    full_links = list(map(lambda x: base_url + x, word_links))
    return full_links


def get_all_word_links():
    print("Getting all word links")
    all_word_links = []
    for letter in alphabet:
        url = base_browse_url.format(letter, 1)
        page_nums = list(range(1, int(get_number_of_pages(url)) + 1))
        for page_num in page_nums:
            all_word_links += get_word_links(base_browse_url.format(letter, page_num))
            print("Letter: {}, Page: {}/{}, Word links collected: {}".format(letter, page_num, max(page_nums), len(all_word_links)))

    return all_word_links


def remove_none_containing_tuple_from_list(l):
    new_l = []
    for tpl in l:
        if None in tpl:
            continue
        else:
            new_l.append(tpl)

    return new_l


def parse_list_of_tuples(l):
    new_l = []
    for tpl in l:
        new_tpl = []
        for elem in tpl:
            elem = elem.string
            try:
                new_tpl.append(elem.strip().replace('"', ""))
            except:
                new_tpl.append(None)
        new_l.append(tuple(new_tpl))

    return new_l


def get_x_y(box_divs):
    x_y = []
    for div in box_divs:
        x_y.append((div.find("p"), div.find("blockquote")))

    return x_y


def get_quotes_from_word_links(all_word_links):
    print("Parsing quotes")
    all_quotes = {}
    for link in all_word_links:
        word = link.split("/")[-2]
        resp = requests.get(link)
        html_doc = resp.text
        soup = bs(html_doc, 'html.parser')
        box_divs = soup.find_all("div", {"class": "box"})

        x_y = get_x_y(box_divs)
        x_y = remove_none_containing_tuple_from_list(x_y)
        x_y = parse_list_of_tuples(x_y)
        x_y = remove_none_containing_tuple_from_list(x_y)
        all_quotes[word] = x_y
        print("{} quote(s) for word: {}".format(len(x_y), word))

    return all_quotes


def save_quotes(quotes):
    json_string = json.dumps(quotes, indent=2)
    with open(save_location, "w") as f:
        f.write(json_string)

if __name__ == "__main__":
    all_word_links = get_all_word_links()
    quotes = get_quotes_from_word_links(all_word_links)
    save_quotes(quotes)
