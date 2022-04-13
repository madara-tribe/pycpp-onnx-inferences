import threading


# 貯金額とする
balance = 0


def change_it(n):
    # ロックを取得
    lock.acquire()
    global balance
    balance = balance + n
    balance = balance - n
    # ロックを解放
    lock.release()


def run_thread(n):
    for i in range(100000):
        change_it(n)


lock = threading.Lock()  # ロックをインスタンス化

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
