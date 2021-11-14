from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory

START_STEP = 66_370_000# start steps when using pretrained model or 0

GAMMA = 0.99           # discount factor
GLOBAL_SEED = 0        # global seed initialize
MEM_SIZE = 100_000     # memory size
RENDER = True         # if true, render gameplay frames 

STACK_SIZE = 4         # stack size

EPS_START = 0.01       # starting epsilon for epsilon-greedy alogrithm
EPS_END = 0.01         # after decay steps, spsilon will reach this and keep
EPS_DECAY = 1_000_000  # steps for epsilon to decay

BATCH_SIZE = 32        # batch size of TD-learning traning value network
POLICY_UPDATE = 4      # policy network update frequency
TARGET_UPDATE = 10_000 # target network update frequency
WARM_STEPS = 50_000    # warming steps before training
MAX_STEPS = 10_000_000 # max training steps
#EVALUATE_FREQ = 100_000
EVALUATE_FREQ = 10_000 # evaluate frequency

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = MyEnv(device)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
    "models/model_6636",
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

# if(START_STEP == 0):
#     os.mkdir("./models")

#### Training ####
obs_queue: deque = deque(maxlen=5)
done = True

# progressive bar
progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=100, leave=False, unit="b")


for step in progressive:
    step += START_STEP
    if done:
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done)

    if step % POLICY_UPDATE == 0 and training:
        agent.learn(memory, BATCH_SIZE)

    if step % TARGET_UPDATE == 0:
        agent.sync()

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
