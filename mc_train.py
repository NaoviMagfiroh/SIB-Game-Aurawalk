"""
Monte Carlo training for 4*4 frozen lake
take about 80s to finish 1000 episodes training
success rate: 79.70%
The shortest route: 6
The longest route: 838

"""
from env import Environment
from agent_brain import MonteCarloTable
import numpy as np
import time


# membuat fungsi untuk menggenarte satu episode 
def monte_carlo_episode():
    episode = []                          # simpan (s,a,r) dari setiap episode
    s = env.reset()                       # pengamatan observasi
    j = 0                                 # jika agen mencapai goal j+=1if 
    t = 0                                 # waktu setiap episode
    start = time.time()                   # catat waktu mulainya

    while True:
        env.render()                      # merefresh environment
        a = RL.choose_action(str(s))      # memilih tindakan berdasarkan pengamatan
        s_, r, done = env.step(a)         # ambil tindakan dan dapatkan pengamatan dan hadiah berikutnya
        episode.append([s, a, r])         # tambahkan (St,At+1,Rt+1) ke dalam array numpy
        s = s_                            # bertukar pengamatan

        # ketika agen mencapai koordinat tujuan
        if env.next_state == [350, 350]:
            j += 1                        # j mencatat waktu mencapai tujuan

        # ketika agen mencapai rintangan atau tujuan
        if done:
            end = time.time()             # catat waktu akhir
            t += end - start              # waktu setiap episode
            env.reset()                   # setel ulang agent
            break
    return episode, t, j


def first_visit(episode):
    episode11 = [i[0] for i in episode]         # ekstrak state  dalam episode
    episode22 = [i[1] for i in episode]         # ekstrak action  dalam episode
    episode33 = [i[2] for i in episode]         # ekstrak reward  dalam episode

    # hapus state yang sama dan tindakan serta imbalannya yang sesuai
    for i in range(len(episode11)-2, 0, -1):
        for j in range(0, i):
            if episode11[i] == episode11[j]:
                del episode11[i]
                del episode22[i]
                del episode33[i-1]
                break
    return episode11, episode22, episode33


#membuat fungsi untuk melakukan training
def train():
    reward_history = []         # rekor pengembalian rata-rata (St,At)
    steps = []                  # rekam langkah setiap episode
    t_ = 0                      # catat jumlah_waktu
    t_sum = []                  # jumlah waktu loop pelatihan
    success_times = 0           # catat waktu mencapai tujuan
    Q_sum = []                  # merekam Q_sum

    # loop untuk semua episode
    for i in range(1000):
        episode, t, j = monte_carlo_episode()
        episode1 = [i[0] for i in episode]   # ekstrak nomor semua langkah
        steps.append(len(episode1))          # menambahkan jumlah state ke dalam daftar

        t_ += t                              # waktu yang dijumlahkan
        t_sum.append(t_)                     # tambahkan jumlah waktu ke dalam daftar untuk plot
        success_times += j                   # waktu sukses mencapaitujuan

        episode11, episode22, episode33 = first_visit(episode)  # first visit MC

        rewards = episode33                  # ekstrak reward
        r = RL.discounted_rewards(rewards)   # menghitung diskon reward
        Q_sum.append(sum(r))                 # menghitung jumlah Q_value G<-gamma*G+R(t+1)

        # loop (perbarui tabel Q) untuk satu episode
        for k in range(len(episode11)):
            RL.update_table(str(episode11[k]), episode22[k], r[k])
        reward_history.append(np.mean(r))               # tambahkan pengembalian rata-rata (St,At) ke dalam daftar

    print('success times:', success_times)              # cetak waktu sukses
    success_rate = success_times / 1000
    print('Success rate: {:.2%}'.format(success_rate))  # menampilkan kesuksesan
    print('running time:', t_)                          # cetak waktu simulasi algoritma

    env.final()                                         # menampilkan rute terakhir

    RL.print_q_table()                                  # menampilkan Q-table

    RL.plot_results(steps, Q_sum, t_sum)                # plot Q_sum dan melangkahi episode


if __name__ == "__main__":
    # memanggil environment
    env = Environment()
    #masukkan tindakan dan status untuk memanggil algoritma utama
    RL = MonteCarloTable(actions=[0, 1, 2, 3],
                         states=['[50.0, 50.0]', '[50.0, 150.0]', '[50.0, 250.0]', '[50.0, 350.0]',
                                 '[150.0, 50.0]', '[150.0, 150.0]', '[150.0, 250.0]', '[150.0, 350.0]',
                                 '[250.0, 50.0]', '[250.0, 150.0]', '[250.0, 250.0]', '[250.0, 350.0]',
                                 '[350.0, 50.0]', '[350.0, 150.0]', '[350.0, 250.0]', '[350.0, 350.0]'])
    env.after(1000, train)
    env.mainloop()
