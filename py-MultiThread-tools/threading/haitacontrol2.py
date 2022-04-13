import threading
import time


def run(n):
    semaphore.acquire()
    time.sleep(1)
    print("current thread: {}\n".format(n))
    semaphore.release()


semaphore = threading.BoundedSemaphore(5)  # 5個のスレッドの同時処理を許容する

for i in range(22):
    t = threading.Thread(target=run, args=("t-{}".format(i),))
    t.start()

while threading.active_count() != 1:
    pass  # print(threading.active_count())
else:
    print('-----全てのスレッドが終了した-----')
