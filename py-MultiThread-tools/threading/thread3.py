import threading
import time


def run(n):
    print("task: {}".format(n))
    time.sleep(1)


for i in range(1, 4):
    t = threading.Thread(target=run, args=("t{}".format(i),))
    t.start()

time.sleep(0.5)
print(threading.active_count())
