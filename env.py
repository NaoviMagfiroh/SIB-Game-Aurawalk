"""
4*4 frozen lake environment with 4 fixed obstacles

Action space:{'up':0  'down':1  'right':2  'left':3}
State space:{'coordinate of tk.canvas': 16 states}
Reward:{'goal':+1  'obstacle':-1  others:0}

"""

import numpy as np
import tkinter as tk
import time
from PIL import Image, ImageTk  

# ukuran environment
pixels = 100         # pixels
env_height = 4       # grid height
env_width = 4        # grid width


#membuat kelas envirronment
class Environment(tk.Tk, object):
    def __init__(self):
        super(Environment, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = 4
        self.title('Monte Carlo. Naovi Magfiroh Ular')
        self.build_environment()

        #Kamus untuk menggambar rute terakhir
        self.d = {}
        self.f = {}

        #Kunci untuk kamus
        self.i = 0

        #Menulis kamus terakhir pertama kali
        self.c = True

        #Menampilkan langkah-langkah untuk rute terpanjang yang ditemukan
        self.longest = 0

        #Menampilkan langkah-langkah untuk rute terpendek
        self.shortest = 0

    #membuat fungsi untuk membangun lingkungan
    def build_environment(self):
        self.canvas = tk.Canvas(self, bg='white', height=env_height * pixels, width=env_width * pixels)

        #Membuat garis kisi
        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas.create_line(x0, y0, x1, y1, fill='grey')
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas.create_line(x0, y0, x1, y1, fill='grey')

        #Membuat variabel untuk obstacle dengan pemanggilan image obstacle
        img_obstacle1 = Image.open("images/obstacle.png")
        img_obstacle2 = Image.open("images/obstacle2.png")
        img_obstacle3 = Image.open("images/obstacle3.png")
        
        self.obstacle1_object = ImageTk.PhotoImage(img_obstacle1)
        self.obstacle2_object = ImageTk.PhotoImage(img_obstacle2)
        self.obstacle3_object = ImageTk.PhotoImage(img_obstacle3)
        
        # Obstacle 1
        self.obstacle1 = self.canvas.create_image(pixels * 1.5, pixels * 1.5, anchor='center',
                                                  image=self.obstacle3_object)
        # Obstacle 2
        self.obstacle2 = self.canvas.create_image(pixels * 3.5, pixels * 1.5, anchor='center',
                                                  image=self.obstacle1_object)
        # Obstacle 3
        self.obstacle3 = self.canvas.create_image(pixels * 0.5, pixels * 3.5, anchor='center',
                                                  image=self.obstacle1_object)
        # Obstacle 4
        self.obstacle4 = self.canvas.create_image(pixels * 3.5, pixels * 2.5, anchor='center',
                                                  image=self.obstacle2_object)
         # Obstacle 5
        self.obstacle5 = self.canvas.create_image(pixels * 1.5, pixels * 3.5, anchor='center',
                                                  image=self.obstacle2_object)
        
        
        

       #Membuat variabel untuk goal dengan pemanggilan image goal
        img_goal = Image.open("images/goal.png")
        self.goal_object = ImageTk.PhotoImage(img_goal)
        self.goal = self.canvas.create_image(pixels * 3.5, pixels * 3.5, anchor='center', image=self.goal_object)

       #Membuat variabel untuk robot dengan pemanggilan image robot
        img_robot = Image.open("images/robot.png")
        self.robot = ImageTk.PhotoImage(img_robot)
        self.agent = self.canvas.create_image(pixels * 0.5, pixels * 0.5, anchor='center', image=self.robot)

        self.canvas.pack()

    #membuat fungsi untuk mengatur ulang lingkungan dan memulai Episode baru
    def reset(self):
        self.update()
        time.sleep(0.001)

        #memperbarui agent
        self.canvas.delete(self.agent)
        self.agent = self.canvas.create_image(pixels * 0.5, pixels * 0.5, anchor='center', image=self.robot)

        #membersihkan kamus dan i
        self.d = {}
        self.i = 0

        #Pengamatan kembali
        return self.canvas.coords(self.agent)

    # Berfungsi untuk mendapatkan pengamatan dan hadiah berikutnya dengan melakukan langkah selanjutnya
    def step(self, action):
        # State agent saat ini
        state = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])

        # Memperbarui state berikutnya sesuai dengan tindakan
        # Action 'up'
        if action == 0:
            if state[1] >= pixels:
                base_action[1] -= pixels
        # Action 'down'
        elif action == 1:
            if state[1] < (env_height - 1) * pixels:
                base_action[1] += pixels
        # Action right
        elif action == 2:
            if state[0] < (env_width - 1) * pixels:
                base_action[0] += pixels
        # Action left
        elif action == 3:
            if state[0] >= pixels:
                base_action[0] -= pixels

       # Memindahkan agent sesuai tindakan
        self.canvas.move(self.agent, base_action[0], base_action[1])

      # Menulis di kamus koordinat rute yang ditemukan
        self.d[self.i] = self.canvas.coords(self.agent)

        # Memperbarui state berikutnya
        self.next_state = self.d[self.i]
        # Memperbarui kunci untuk kamus
        self.i += 1

       # Menghitung hadiah untuk agent
        if self.next_state == self.canvas.coords(self.goal):
            reward = 1
            done = True

        # Mengisi kamus pertama kali
            if self.c:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                self.c = False
                self.longest = len(self.d)
                self.shortest = len(self.d)

           # Memeriksa apakah rute yang ditemukan saat ini lebih pendek
            if len(self.d) < len(self.f):
               # Menyimpan jumlah langkah untuk rute terpendek
                self.shortest = len(self.d)
                # Membersihkan kamus untuk rute terakhir
                self.f = {}
                # Menugaskan ulang kamus
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]

            
            # Menyimpan jumlah langkah untuk rute terpanjang
            if len(self.d) > self.longest:
                self.longest = len(self.d)

        elif self.next_state in [self.canvas.coords(self.obstacle1),
                                 self.canvas.coords(self.obstacle2),
                                 self.canvas.coords(self.obstacle3),
                                 self.canvas.coords(self.obstacle4),
                                 self.canvas.coords(self.obstacle5)]:
            reward = -1
            done = True

            # Membersihkan kamus dan i
            self.d = {}
            self.i = 0

        else:
            reward = 0
            done = False

        return self.next_state, reward, done

   # Berfungsi untuk menyegarkan lingkungan
    def render(self):
        time.sleep(0.001)
        self.update()

   # Fungsi untuk menunjukkan rute yang ditemukan
    def final(self):
       # Menghapus agen di akhir
        self.canvas.delete(self.agent)

        # Menampilkan jumlah langkah
        print('The shortest route:', self.shortest)
        print('The longest route:', self.longest)

        # Membuat titik awal
        self.initial_point = self.canvas.create_oval(40, 40, 60, 60, fill='red', outline='red')

        # Mengisi rute
        for j in range(len(self.f)):
            # Menampilkan koordinat rute akhir
            print(self.f[j])
            self.track = self.canvas.create_oval(
                self.f[j][0] - 10, self.f[j][1] - 10,
                self.f[j][0] + 10, self.f[j][1] + 10,
                fill='red', outline='red')



# Ini menunjukkan lingkungan statis tanpa menjalankan algoritma
if __name__ == '__main__':
    env = Environment()
    env.mainloop()
