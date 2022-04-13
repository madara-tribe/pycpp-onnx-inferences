import time, sys
import numpy as np
from multiprocessing import Pool
import multiprocessing

def cos_sim(v1, v2):
    v1_ = np.array(v1)
    v2_ = np.array(v2)
    return np.dot(v1_, v2_) / (np.linalg.norm(v1_) * np.linalg.norm(v2_))

class Sample:
    def __init__(self, user_size=943, item_size=1682, file_path="ml-100k/u.data", pool=True):
        self.file_path = file_path
        # user数×アイテム数のリスト
        self.eval_table = [[0 for _ in range(item_size)] for _ in range(user_size)]
        # user数×user数のcos類似度テーブル
        self.sim_table = [[0 for _ in range(user_size)] for _ in range(user_size)]
        
        self.pool = pool
    def distinguish_info(self, line):
        u_id, i_id, rating, timestamp = line.replace("\n", "").split("\t")
        # u_idとi_idはitemのindexを一つずらす
        return int(u_id)-1, int(i_id)-1, float(rating), timestamp


    def calc_cossim(self, target_u_id, target_user_eval):
        for u_id ,user_eval in enumerate(self.eval_table):
            self.sim_table[target_u_id][u_id] = cos_sim(target_user_eval, user_eval)
        if self.pool:
            return self.sim_table[target_u_id]
    def wrapper(self, args):
        return self.calc_cossim(*args)
        
    def run(self):
        f = open(self.file_path , 'r')
        # userとitemのテーブル作成
        start = time.time()
        for line in f:
            u_id, i_id, rating, _ = self.distinguish_info(line)
            self.eval_table[u_id][i_id] = rating

        # テーブルに基づいてcos類似度作成
        for target_u_id, target_user_eval in enumerate(self.eval_table):
            self.calc_cossim(target_u_id, target_user_eval)
        times = time.time()-start
        print("total time is:{}".format(times))
        
    def run_pool(self, processes=8):
        f = open(self.file_path , 'r')
        # userとitemのテーブル作成
        start = time.time()
        for line in f:
            u_id, i_id, rating, _ = self.distinguish_info(line)
            self.eval_table[u_id][i_id] = rating

        # テーブルに基づいてcos類似度作成
        tmp = [(target_u_id, target_user_eval) for target_u_id, target_user_eval in enumerate(self.eval_table)]
        with Pool(processes=processes) as pool:
            # 変更
            self.right_sim_table = pool.map(self.wrapper, tmp)
        times = time.time()-start
        print("total time is :{}".format(times))


if __name__ == "__main__":
    num_cpu = multiprocessing.cpu_count()
    pool = str(sys.argv[1])
    if pool=='pool':
        s = Sample(pool=True)
        s.run_pool(processes=num_cpu)
    else:
        s = Sample(pool=None)
        s.run()
