import threading
import time
from random import randint
from collections import deque


class Producer(threading.Thread):
    def run(self):
        global stocks
        while True:
            if lock_con.acquire():
                products = [randint(0, 100) for _ in range(5)]
                stocks.extend(products)
                print('生産者{}は{}を生産した。'.format(self.name, stocks))
                lock_con.notify()
                lock_con.release()
            time.sleep(3)


class Consumer(threading.Thread):
    def run(self):
        global stocks
        while True:
            lock_con.acquire()
            if len(stocks) == 0:
                # 商品が無くなったら生産されるまで待つ
                # notfifyされるまでスレッドをハングアップ
                lock_con.wait()
            print('お客様{}は{}を買った。在庫: {}'.format(self.name, stocks.popleft(), stocks))
            lock_con.release()
            time.sleep(0.5)


stocks = deque()
lock_con = threading.Condition()
p = Producer()
c = Consumer()
p.start()
c.start()
