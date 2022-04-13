import threading

num = 4


def start_game():
    print('{}人になったため、ゲーム開始。'.format(num))


lock = threading.Lock()
barrier = threading.Barrier(num, action=start_game)


class Player(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            if not barrier.broken:
                print('{}さんが参加しました。'.format(self.name))
                barrier.wait(2)
        except threading.BrokenBarrierError:
            print('ゲーム開始できないため、{}が退出しました。'.format(self.name))


players = []
for i in range(10):
    p = Player(name='Player {}'.format(i))
    players.append(p)

for p in players:
    p.start()
