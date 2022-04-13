from multiprocessing import Queue, Process
import random
import time

# produce Process
def produce(queue):
  for i in range(10):
    queue.put(i)
    time.sleep(random.randint(1, 5))

# consume Process
def consume(queue):
  for i in range(10):
    n = queue.get()
    print(n)
    time.sleep(random.randint(1, 5))

if __name__ == '__main__':
  queue = Queue()
  
  p0 = Process(target=produce, args=(queue,))
  p1 = Process(target=produce, args=(queue,))
  c0 = Process(target=consume, args=(queue,))
  c1 = Process(target=consume, args=(queue,))
  
  p0.start()
  p1.start()
  c0.start()
  c1.start()
  
  # wait for process finish
  p0.join()
  p1.join()
  c0.join()
  c1.join()
