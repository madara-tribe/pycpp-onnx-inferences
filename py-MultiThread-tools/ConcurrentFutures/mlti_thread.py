import concurrent.futures
import urllib.request
import time, sys, os

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://some-made-up-domain.com/']

# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

class ConcurrentFutures():
    def get_detail(self):
        # Start the load operations and mark each future with its URL
        for url in URLS:
            try:
                data = load_url(url,60)
            except Exception as exc:
                print(f'{url} generated an exception: {exc}')
            else:
                print(f'{url} page is len(data) bytes')
    
    def mlti_thread_get_detail(self):
        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(f'{url} generated an exception: {exc}')
                else:
                    print(f'{url} page is len(data) bytes')

def main():
    pool = str(sys.argv[1])
    CFthread = ConcurrentFutures()
    startTime = time.time()
    if pool=='pool':
        print('multi thread')
        CFthread.mlti_thread_get_detail()
    else:
        print('no thread')
        CFthread.get_detail()
    runTime = time.time() - startTime
    print (f'Time:{runTime}[sec]')

if __name__ == '__main__':
    main()
