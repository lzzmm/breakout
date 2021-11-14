import numpy as np
import matplotlib.pyplot as plt


data1 = []
data2 = []
rewards = []
rewards_d = []
with open('dqn-breakout/rewards.txt', 'r') as f:
    for line in f:
        data1.append([i for i in line.split()])

with open('dueling-dqn-breakout/rewards.txt', 'r') as f:
    for line in f:
        data2.append([i for i in line.split()])

# print(len(data2))

# DATANUM1 = len(data1)
# X1 = DATANUM1/AVG
# DATANUM2 = len(data2)
# X2 = DATANUM2/AVG

X1 = 0
X2 = 6666
Y1 = 0
Y2 = 432
AVG = 100  # (X2-X1)/200

tmp = 0
count = 0
for i in data1:
    # rewards.append(float(i[2]))
    tmp += float(i[2])
    count = count+1
    if count % AVG == 0:
        rewards.append(tmp/AVG)
        tmp = 0

tmp = 0
count = 0
for j in data2:
    # rewards_d.append(float(j[2]))
    tmp += float(j[2])
    count = count+1
    if count % AVG == 0:
        rewards_d.append(tmp/AVG)
        tmp = 0

DATANUM1 = len(rewards)*AVG
DATANUM2 = len(rewards_d)*AVG

# print(rewards)
t1 = np.arange(0, DATANUM1, AVG)
t2 = np.arange(0, DATANUM2, AVG)

plt.figure(dpi=256)

line_dqn, = plt.plot(t1, rewards, 'g-')
line_duel, = plt.plot(t2, rewards_d, 'b-')

plt.axis([X1, X2, Y1, Y2])

plt.legend(handles=[line_dqn, line_duel], labels=["DQN", "Dueling DQN"], loc="lower right")
plt.xlabel('Episodes($\\times 10^4 $)')
plt.ylabel('Average Reward per Episode')
plt.title('Average Reward on Breakout')

# plt.savefig('plot.png')
plt.show()

# d1s = 0
# d2s = 0
# for t in range(0, 2000):
#     d1s += rewards[t]
#     d2s += rewards_d[t]
    
# print(d2s/d1s - 1.0)

