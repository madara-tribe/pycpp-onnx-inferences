import threading
import time


def run(n):
    print("task: {}".format(n))
    time.sleep(1)
    print('3')
    time.sleep(1)
    print('2')
    time.sleep(1)
    print('1')


for i in range(1, 4):
    t = threading.Thread(target=run, args=("t{}".format(i),))
    # setDaemon(True)
    t.setDaemon(True) 
    t.start()

time.sleep(1.5)
print('スレッド数: {}'.format(threading.active_count()))
