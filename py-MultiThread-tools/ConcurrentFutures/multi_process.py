import math, time
import sys
import concurrent.futures

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return '0'
    return '{}'.format(n)

class MultiProcess():
    def run(self):
        for number, prime in zip(PRIMES, map(is_prime, PRIMES)):
            print(f'{number} is prime: {prime}')
    
    def multi_precoss_run(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
                print(f'{number} is prime: {prime}')
    

if __name__ == '__main__':
    multi = str(sys.argv[1])
    MULTI = MultiProcess()
    startTime = time.time()
    if multi=='m':
        print('multi process')
        MULTI.multi_precoss_run()
    else:
        print('No multi process')
        MULTI.run()
    runTime = time.time() - startTime
    print(f'Time:{runTime}[sec]')

