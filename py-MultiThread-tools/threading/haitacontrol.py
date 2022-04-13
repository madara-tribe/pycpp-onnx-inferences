import threading


# 貯金額とする
balance = 0


def add_it(n):
    lock.acquire()
    global balance
    balance = balance + n
    return balance


def sub_it(n):
    lock.acquire()
    global balance
    balance = balance - n
    return balance


def change_it(n):
    # ロックを取得
    lock.acquire()
    global balance
    balance = add_it(n)
    balance = sub_it(n)
    # 再帰的ににロックを解放
    lock.release()


def run_thread(n):
    for i in range(1000):
        change_it(n)


lock = threading.RLock()  # ロックをインスタンス化

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
