import time
from multiprocessing import Queue, Process

def f2(q):
    time.sleep(3)
    q.put([42, None, "Hello"])

if __name__ == "__main__":
    q = Queue()
    # キューを引数に渡して、サブプロセスを作成
    p = Process(target=f2, args=(q,))
    p.start()
    # wqait for queue get()
    print(q.get())
    p.join()
