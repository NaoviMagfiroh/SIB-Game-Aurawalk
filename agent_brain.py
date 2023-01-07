"""
Brain of the RL agent
including the three algorithms (Monte Carlo)

Hyper-parameters setting:
learning rate: 0.1
discount factor: 0.9
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Mendefinisikan kelas Reinforcement 
class RL:
    def __init__(self, actions, states):
        self.states = states       # state space
        self.actions = actions     # action space
        self.LearningRate = 0.1    # learning rate
        self.gamma = 0.9           # discount factor
        self.epsilon = 0.1         # e-greedy parameter

        # Membuat sebuat variabel q_table(dengan menggunakan 16 states dan 4 action)
        self.q_table = pd.DataFrame(np.zeros((16, 4)), index=self.states, columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        #Memilih tindakan sesuai dengan algoritma soft e-greedy

        # 1-e+e/|A(St)|  90% + 2.5% untuk memilih q_value terbaik
        if np.random.uniform(0, 1) > self.epsilon + self.epsilon / 4:
            state_action = self.q_table.loc[observation, :]  #diambil dari daftar 4 action dalam state

            # jika Q_values sama maka pilih tindakan acak
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()  #pilih nilai q_value yang paling maksimal
        else:
            action = np.random.choice(self.actions)  #kiri 7,5% untuk memilih secara acak
        return action

    def print_q_table(self):
        #mengatur parameter table
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.unicode.ambiguous_as_wide', True)
        pd.set_option('display.unicode.east_asian_width', True)
        print()
        print('Q-table:')
        print(self.q_table)

    def plot_results(self, steps, q_sum, t_sum):
        #Plot langkah-langkah di atas episode
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'blue', linewidth=1)
        plt.title('Steps over episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Steps')

        #Plot Q_sum di atas episode
        plt.figure()
        plt.plot(np.arange(len(q_sum)), q_sum, 'pink', linewidth=1)
        plt.title('Sum of Q_value over episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of Q_value')

        #Plot Episode dari waktu ke waktu
        plt.figure()
        plt.plot(t_sum, np.arange(len(steps)), 'green')
        plt.title('Episodes over Time')
        plt.xlabel('Time(s)')
        plt.ylabel('Episodes')

        plt.show()


# -----MonteCarlo Algorithm-----

class MonteCarloTable(RL):
    def __init__(self, actions, states):
        super(MonteCarloTable, self).__init__(actions, states)

    #menghitung pengembalian diskon G
    def discounted_rewards(self, rewards):
        current_reward = 0
        discounted_rewards = [0] * len(rewards)
        for t in reversed(range(len(rewards))):
            current_reward = self.gamma * current_reward + rewards[t]
            discounted_rewards[t] = current_reward
        return discounted_rewards

    def update_table(self, s, a, reward_discounted):
        #Keadaan saat ini di posisi saat ini
        q = self.q_table.loc[s, a]

        #Memperbarui tabel-Q dengan pengetahuan baru
        self.q_table.loc[s, a] += self.LearningRate * (reward_discounted - q)

        return self.q_table.loc[s, a]
