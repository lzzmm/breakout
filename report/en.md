## DCS245 <br>Reinforcement Learning and Game Theory<br>2021 Fall

# Mid-term Assignment

#### `19335025 Chen Yuhan`<br>`19335026 Chen Yuyan`





## 0 Background

**Breakout** is an arcade game developed and published by Atari, Inc., and released on May 13, 1976. Breakout begins with eight rows of bricks, with each two rows a different kinds of color. The color order from the bottom up is yellow, green, orange and red. Using a single ball, the player must knock down as many bricks as possible by using the walls and/or the paddle below to hit the ball against the bricks and eliminate them. If the player's paddle misses the ball's rebound, they will lose a turn. The player has three turns to try to clear two screens of bricks. Yellow bricks earn one point each, green bricks earn three points, orange bricks earn five points and the top-level red bricks score seven points each. 

## 1 Introduction

### 1.1 Reinforcement Learning



### 1.2 Markov Decision Process



### 1.3 Q-Learning



### 1.4 DQN (NIPS 2013)



### 1.5 Nature-DQN



## 2 `dqn-breakout`[$^1$](https://gitee.com/goluke/dqn-breakout) Analysis

\[1\]: Base implementation: https://gitee.com/goluke/dqn-breakout

### 2.1 `main.py`

`mian.py` 是整个程序的入口，它首先定义了这些常量：

```c
START_STEP = 0         # start steps when using pretrained model

GAMMA = 0.99           # discount factor
GLOBAL_SEED = 0        # global seed initialize
MEM_SIZE = 100_000     # memory size
RENDER = False         # if true, render gameplay frames 

STACK_SIZE = 4         # stack size

EPS_START = 1          # starting epsilon for epsilon-greedy alogrithm
EPS_END = 0.05         # after decay steps, spsilon will reach this and keep
EPS_DECAY = 1_000_000  # steps for epsilon to decay

BATCH_SIZE = 32        # batch size of TD-learning traning value network
POLICY_UPDATE = 4      # policy network update frequency
TARGET_UPDATE = 10_000 # target network update frequency
WARM_STEPS = 50_000    # warming steps before training
MAX_STEPS = 50_000_000 # max training steps
EVALUATE_FREQ = 10_000 # evaluate frequency
```

- `START_STEP` 是模型开始训练的步数，方便使用已有的模型继续计算。没有使用预先训练好的模型开始计算时为 `0`；

- `GAMMA` 是折扣（衰减）因子 $\gamma$ ，设为 `0.99`；
- `MEM_SIZE` 是 `ReplayMemory` 中的 `capacity`；
- `RENDER` 为 `TRUE` 的时候在每次评价的时候都会渲染游戏画面；
- `STACK_SIZE` 是 `ReplayMemory` 中的 `channels`；
- `EPS_START` 和 `EPS_END` 是在 `EPS_DECAY` 步中 $\epsilon $ 衰减的开始和结尾值，之后 $\epsilon$ 一直保持在 `EPS_END` ，值得一提的是一开始 `EPS_START` 会是 `1` ，但是后面加载模型继续训练的时候有必要更改成较小的数值，否则加载的模型的性能不能很好地表现；
- `BATCH_SIZE` 是在从 `ReplayMemory` 中取样的时候的取样个数；
- `POLICY_UPDATE` 是策略网络更新的频率；
- `TARGET_UPDATE` 是目标网络更新的频率；
- `WARM_STEPS` 是为了等到 `ReplayMemory` 中有足够的记录的时候再开始降低 $\epsilon$ ；
- `MAX_STEPS` 是训练的步数；
- `EVALUATE_FREQ` 是评价的频率。

接着初始化随机数，初始化计算设备，初始化环境 `MyEnv`、智能体 `Agent` 和 `ReplayMemory`。

注意此处把 `done` 置为 `True` 是为了开始训练时初始化环境并记录一开始的观察。

然后开始实现上面所说的 **Nature DQN** 算法，在循环中首先判断一个回合是否已经结束，若结束则重置环境状态并将观察数据入队存储：

```python
if done:
    observations, _, _ = env.reset()
    for obs in observations:
        obs_queue.append(obs)
```

接着判断是否已经经过 `Warming steps` ，若是，则将 `training` 置为 `True` ，此时则会开始衰减 $\epsilon$：

```python
training = len(memory) > WARM_STEPS
```

接着观察现在的状态 `state`，并根据状态选择动作 `action`，然后获得观察到的新的信息 `obs`、反馈 `reward` 和是否结束游戏的状态 `done`：

```python
state = env.make_state(obs_queue).to(device).float()
action = agent.run(state, training)
obs, reward, done = env.step(action)
```

把观察入队，把当前状态、动作、反馈、是否结束都记录入 `MemoryReplay`：

```python
obs_queue.append(obs)
memory.push(env.make_folded_state(obs_queue), action, reward, done)
```

更新策略网络和同步目标网络，同步目标网络就是把目标网络的参数更新为策略网络的参数：

```python
if step % POLICY_UPDATE == 0 and training:
    agent.learn(memory, BATCH_SIZE)
if step % TARGET_UPDATE == 0:
    agent.sync()
```

评价当前网络，将平均反馈和训练出来的策略网络保存，并结束游戏。若 `RENDER` 为 `True` 则渲染游戏画面：

```python
if step % EVALUATE_FREQ == 0:
    avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
    with open("rewards.txt","a") as fp:
        fp.write(f"{step//EVALUATE_FREQ:4d} {step:8d} {avg_reward:.1f}\n")
    if RENDER:
        prefix = f"eval/eval_{step//EVALUATE_FREQ:04d}"
        os.mkdir(prefix)
        for ind, frame in enumerate(frames):
            with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                frame.save(fp, format="png")
    agent.save(f"models/model_{step//EVALUATE_FREQ:04d}")
    done = True
```



### 2.2 `utils_drl.py`

`utils_drl.py` 中实现了智能体 `Agent` 类，初始化了传入参数和两个模型，并且在没有传入训练好的模型的时候初始化模型参数，在传入训练好的模型的时候加载模型参数。

```python
if restore is None:
    self.__policy.apply(DQN.init_weights)
    else:
        self.__policy.load_state_dict(torch.load(restore))
    self.__target.load_state_dict(self.__policy.state_dict())
    self.__optimizer = optim.Adam(
        self.__policy.parameters(),
        lr=0.0000625,
        eps=1.5e-4,
    )
    self.__target.eval()
```

`Agent` 类中定义了四个函数，分别如下：

1. `run()` 函数实现了通过 $\epsilon - greedy$ 策略根据现在的状态选择一个动作；

2. `learn()` 函数实现了更新神经网络的参数：

   ```python
   def learn(self, memory: ReplayMemory, batch_size: int) -> float:
       """learn trains the value network via TD-learning."""
       # 从memory中随机取样本
       state_batch, action_batch, reward_batch, next_batch,
       done_batch = \
       memory.sample(batch_size)
       # state预期的values
       values = self.__policy(state_batch.float()).gather(1,
       action_batch)
       # 下一个state预期的values
       values_next =
       self.__target(next_batch.float()).max(1).values.detach()
       # state现实的values=衰减因子gamma*values_next+reward
       expected = (self.__gamma * values_next.unsqueeze(1)) * \
       (1. - done_batch) + reward_batch
       # 损失
       loss = F.smooth_l1_loss(values, expected)
       self.__optimizer.zero_grad()
       # 求导
       loss.backward()
       for param in self.__policy.parameters():
       param.grad.data.clamp_(-1, 1)
       #更新policy神经网络的参数
       self.__optimizer.step()
   	return loss.item()
   ```

3. `sync()` 函数将目标网络延时更新为策略网络；

4. `save()` 函数保存当前的策略网络参数。



### 2.3 `utils_env.py`

`utils_env.py` 主要实现了调用包并配置运行的游戏环境，主要的几个函数如下：

1. `reset()` 初始化游戏并提供5步的时间让智能体观察环境；
2. `step()` 执行一步动作，返回最新的观察，反馈和游戏是否结束的布尔值；
3. `evaluate()` 使用给定的智能体模型来运行游戏并返回平均反馈值和记录游戏的帧。

### 2.4 `utils_model.py`

`utils_model.py` 中使用 `pytorch` 实现了 Nature-DQN 模型。

### 2.5 `utils_memory.py`

`utils_memory.py` 中主要是 `class ReplayMemory` 的实现。主要实现了数据存储和随机抽样。 

## 3 Dueling DQN

Dueling DQN 考虑将 Q 网络分成两部分，第一部分是仅仅与状态 $S$ 有关，与具体要采用的动作 $A$ 无关，这部分我们叫做价值函数部分，记做$V(S,w,α)$，第二部分同时与状态状态 $S$ 和动作 $A$ 有关，这部分叫做优势函数(Advantage Function)部分,记为 $A(S,A,w,β)$ ，那么最终我们的价值函数可以将原来的 $Q(S,A;w)$ 替换为
$$
V(S;w,a)+(A(S,A;w,\beta)-\frac{1}{\mathcal{A}}\sum_{a'\in |\mathcal{A}|}A(a,a';w,\beta))
$$
$w$ 是公共部分参数，$\alpha$ 是 $V$ 独有的参数，$\beta$ 是 $A$ 独有的参数。

![image-20211114105819228](img/en/image-20211114105819228.png)

```python
class Dueling-DQN(nn.Module):
    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1_a = nn.Linear(64*7*7, 512)
        self.__fc1_v = nn.Linear(64*7*7, 512)
        self.__fc2_a = nn.Linear(512, action_dim)
        self.__fc2_v = nn.Linear(512, 1)
        self.__act_dim = action_dim
        self.__device = device

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        xv = x.view(x.size(0), -1)
        vs = F.relu(self.__fc1_v(xv))
        vs = self.__fc2_v(vs).expand(x.size(0), self.__act_dim)
        asa = F.relu(self.__fc1_a(xv))
        asa = self.__fc2_a(asa)
 
        return vs + asa - asa.mean(1).unsqueeze(1).expand(x.size(0),self.__act_dim)
    
```



## 4 Experiments

我们的实验在前一百万次把 $\epsilon$ 从 `1` 降低到 `0.1`，又花了一百万次降低到 `0.01`。以下是我们对实验结果的分析。

![fig1](img/en/Figure_1.png)

*Figure 1: Sampled every $10^6$ episodes,  $6666 \times 10^4$  episodes.*

首先从这张平滑过后的平均反馈图可以看到，在前一千六百万次左右的训练中，Dueling-DQN 大部分时间要比 DQN 高，说明训练的比较快。在四千万次训练之前，DQN 一直保持低于 Dueling-DQN 的反馈。到了三千五百万次到四千万次左右训练的时候，DQN 才又开始增长。四千万次训练之后 DQN 的平均反馈值略高于 Dueling-DQN 的平均反馈值。不过这也和 Dueling-DQN 论文中的结果相似，他们在测试 Breakour 的时候确实最后结果不如 DQN，但是拟合速度略快，我们推测这是由于 Breakout 中的动作比较少，所以提升不明显。



![fig2](img/en/Figure_2.png)

*Figure 1: Sampled every $10^5$ episodes, $6666 \times 10^4$ episodes.*

这张图是没有做过多平滑的，可以看出拟合之前 Dueling-DQN 波动较大，拟合后波动比 DQN 小一点。

我们统计得知，他们在下面三个阶段 Dueling-DQN 的平均反馈值相对于 DQN 的平均反馈值的提升如下：

| Episodes                          | 0 - 2000 | 2000 - 4000 | 4000 - 6000 |
| --------------------------------- | -------- | ----------- | ----------- |
| Avg. Reward: Dueling-DQN/DQN-100% | 14.6563% | 22.9607%    | -0.7542%    |

![image-20211114112548952](img/en/image-20211114112548952.png)

下面两张图是他们拟合前的放大。

![fig3](img/en/Figure_3.png)

*Figure 3: DQN's average reward per every $5 \times 10^4$ episodes, $2000 \times 10^4$ episodes.*



![fig4](img/en/Figure_4.png)

*Figure 4: Dueling-DQN's average reward per every $5 \times 10^4$ episodes, $2000 \times 10^4$ episodes.*

## 5 Summary

### 5.1 Summary



### 5.2 Open Source Repository

Our code and report are open source at [lzzmm/breakout](https://github.com/lzzmm/breakout).

### 5.3 Authorship

| Name   | ID         | Ideas(%) | Coding(%) | Writing(%) |
| ------ | ---------- | -------- | --------- | ---------- |
| 陈禹翰 | `19335025` | `0%`     | `0%`      | `0%`       |
| 陈煜彦 | `19335026` | `0%`     | `0%`      | `0%`       |

## References

[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602v1)

[Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581v3)



