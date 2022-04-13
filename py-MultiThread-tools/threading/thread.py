import threading
import time


def run(n):
    # threading.current_thread().nameはgetName()を呼び出す
    print("task: {} (thread name: {})".format(n, threading.current_thread().name))
    time.sleep(1)
    print('2s')
    time.sleep(1)
    print('1s')
    time.sleep(1)
    print('0s')
    time.sleep(1)


t1 = threading.Thread(target=run, args=("t1",))
t2 = threading.Thread(target=run, args=("t2",), name='Thread T2') # ここではsetName()が呼び出される
# start()
t1.start()
t2.start()
# join()
t1.join()
t2.join()
# join()を呼び出したため
# メインスレッドは上記のスレッドが終わるまで待機し
# 全部終わったらprintする
print(threading.current_thread().name)
