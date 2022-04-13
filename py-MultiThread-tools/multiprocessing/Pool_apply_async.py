import time, sys, os
from multiprocessing import Pool, Process

def nijou(inputs):
    x = inputs
    print('input: %d' % x)
    time.sleep(2)
    retValue = x * x
    print('double: %d' % retValue)
    return(retValue)

class PoolApply:
    def __init__(self, processes):
        self.processes = processes
    def pool_apply(self):
        p = Pool(self.processes)
        stime = time.time()
        values = [x for x in range(10)]
        #print(values)
        # not list
        #for x in range(10):
        result = p.apply(nijou, args=[values[9]])
        print(result)
        print('time is ', time.time() -stime)
        p.close()
        
    def pool_apply_async(self):
        p = Pool(self.processes)
        stime = time.time()
        # プロセスを2つ非同期で実行
        values = [x for x in range(10)]
        result = p.apply_async(nijou, args=[values[9]])
        result2 = p.apply_async(nijou, args=[values[9]])
        print(result.get())
        print(result2.get())
        print('time is ', time.time() -stime)
        p.close()
        
if __name__ == "__main__":
    case_no = int(sys.argv[1])
    num_process = int(sys.argv[2])
    pool = PoolApply(num_process)
    if case_no==1:
        pool.pool_apply()
    elif case_no==2:
        pool.pool_apply_async()
    
    
