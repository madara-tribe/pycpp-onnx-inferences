import time, os, sys
from multiprocessing import Pool
import multiprocessing

print("start worker={}", os.getpid())

def nijou(inputs):
    x = inputs
    print('input: %d' % (x))
    time.sleep(2)
    retValue = x * x
    print('double: %d' % (retValue))
    return(retValue)

if __name__ == "__main__":
    num_cpu = multiprocessing.cpu_count()
    case = int(sys.argv[1])
    values = [x for x in range(10)]
    if case==1:
        with Pool(processes=num_cpu) as p:
            print('case1')
            stime = time.time()
            print(values)
            # list is required
            result = p.map(nijou, values)
    else:
        with Pool(processes=num_cpu) as p:
            print('case2')
            stime = time.time()
            print(values)
            # list is required
            result = p.map(nijou, values)
    print(result)
    print('time is ', time.time() -stime)
