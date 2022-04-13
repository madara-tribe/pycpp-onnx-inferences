from multiprocessing import Lock, Process

def f4(lock, i):
    # ロックを取得します.
    lock.acquire()
    # ロック中は、他のプロセスやスレッドがロックを取得できません（ロックが解放されるまで待つ）
    try:
        print('Hello', i)
    finally:
        # ロックを解放します.
        lock.release()

if __name__ == "__main__":
    # ロックを作成します.
    lock = Lock()
    for num in range(10):
        Process(target=f4, args=(lock, num)).start()
