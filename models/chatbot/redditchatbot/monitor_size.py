import os
import time

begin = time.time()
while 1:
    os.system("ls -la | grep 2014-01_-_2017-10.db")
    print("Time elapsed: {}".format(time.time() - begin))
    time.sleep(1)
