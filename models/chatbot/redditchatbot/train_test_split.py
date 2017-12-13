import sqlite3
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


startDate = "2014-01"
endDate = "2017-09"

def generate_dates():
    start = datetime.strptime(startDate, '%Y-%m').date()
    end = datetime.strptime(endDate, '%Y-%m').date()
    all_dates = [startDate]
    cur_date = start

    while cur_date < end:
        cur_date += relativedelta(months=1)
        all_dates.append("-".join(str(cur_date).split("-")[:-1]))

    return all_dates


if __name__ == "__main__":
    timeframes = generate_dates()

    for timeframe in timeframes:
        connection = sqlite3.connect('{}.db'.format(timeframe))
        c = connection.cursor()
        limit = 5000
        last_unix = 0
        cur_length = limit
        counter = 0
        test_done = False

        while cur_length == limit:

            df = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix,limit),connection)
            last_unix = df.tail(1)['unix'].values[0]
            cur_length = len(df)

            if not test_done:
                with open('test.from','a', encoding='utf8') as f:
                    for content in df['parent'].values:
                        f.write(content+'\n')

                with open('test.to','a', encoding='utf8') as f:
                    for content in df['comment'].values:
                        f.write(str(content)+'\n')

                test_done = True

            else:
                with open('train.from','a', encoding='utf8') as f:
                    for content in df['parent'].values:
                        f.write(content+'\n')

                with open('train.to','a', encoding='utf8') as f:
                    for content in df['comment'].values:
                        f.write(str(content)+'\n')

            counter += 1
            if counter % 20 == 0:
                print(counter*limit,'rows completed so far')
