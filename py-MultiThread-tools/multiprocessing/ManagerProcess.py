from multiprocessing import Manager, Process

def f6(d, l):
    # 辞書型に値を詰め込みます.
    d[1] = '1'
    d["2"] = 2
    d[0.25] = None
    # 配列を操作します（ここでは逆順に）.
    l.reverse()

if __name__ == "__main__":
    # マネージャーを生成します.
    with Manager() as manager:
        # マネージャーから辞書型を生成します.
        d = manager.dict()
        # マネージャーから配列を生成します.
        l = manager.list(range(10))
        # サブプロセスを作り実行します.
        p = Process(target=f6, args=(d,l))
        p.start()
        p.join()
        # 辞書からデータを取り出します.
        print(d)
        # 配列からデータを取り出します.
        print(l)
