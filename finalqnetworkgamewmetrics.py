import math
import random
import pygame
import pygame.font
import json
from collections import namedtuple, deque
from itertools import count
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
counter = 0
q_values1 = [0.0, 0.0, 0.0]
high_score = 0
GAME_SPEED = 1.5
window = pygame.display.set_mode((800, 600))
def write_text(text, position):
    message = Font.render(text, False, (0,0,0))
    win.blit(message, position)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = (self.fc3(x))
        return x

# QAgent class is one instance of an agent running in a Deep Q-Learning Model
# - Is able to select an action based on a state
# - Is able to update its own neural network using a state and an action
# - 
class QAgent:

    def __init__(self, state_dim, action_dim, learning_rate, discount_factor, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.alpha = 0.99
        self.alpha_decay = 0.9999999
        self.epsilon = epsilon

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.SGD(self.q_network.parameters(), lr = learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        global q_values1
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            q_values1 = q_values.squeeze().tolist()
            #write_text(str(q_values), (0,0))
            return torch.argmax(q_values).item()

    def update_q_values(self, state, action, next_state, reward, done):

        if reward != 0:
            print("hi!")
        # turn the state into a state tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # turn next state into a state tensor
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # turn the action into a tensor
        action_tensor = torch.LongTensor([action])

        # get the q_values of the original state
        q_values = self.q_network(state_tensor)

        # select the q_value of the action that was taken
        q_value = q_values[0][action_tensor]

        # get the q_values of the new state
        next_q_values = self.target_network(next_state_tensor).detach()
        
        # determine the maximum q_value that could result from this state
        max_next_q_value = next_q_values.max(1)[0]

        # determine the target Q_value
        expected_q_value = q_value + self.alpha * (reward + self.discount_factor * max_next_q_value - q_value)
        self.alpha *= self.alpha_decay

        expected_q_values = q_values.detach().clone()
        
        expected_q_values[0][action] = expected_q_value

        self.optimizer.zero_grad()
        #loss = self.loss_fn(q_values, expected_q_value.unsqueeze(1))
        loss = self.loss_fn(q_value, expected_q_value)
        loss.backward()
        self.optimizer.step()
        self.update_epsilon()

    def update_epsilon(self):
        self.epsilon *= 0.99999

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# modified to only hold the STATE and ACTION
# Instead of holding the "next state", that state is just going to be 
# whatever the agent was in when it receives a reward or punishment
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class Box:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height

class Camera:
    def __init__(self, left, top, width, height):
        self.rect = Box(left, top, width, height)
    
    # follow a specific player
    def follow(self, player):
        self.rect.top = player.rect.top + (self.rect.height / 2)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()

        self.rect = Box(100, 200, 200, 100)
        self.highest_platform_index = 0
        self.player_vel = 1 * GAME_SPEED
        self.vert_vel = 0
        self.jump_vel = 0.7 * GAME_SPEED
        self.gravity = 0.0005 * GAME_SPEED * GAME_SPEED
        self.can_jump = False
        self.is_alive = True

    def update(self):
        self.draw_player()
        status = self.platform_collision(pm.platforms)
        if not self.can_jump:
            self.vert_vel -= self.gravity
        self.rect.top += self.vert_vel
        if self.rect.top < pm.platforms[self.highest_platform_index].rect.top:
            self.is_alive = False
        
        return status

    def platform_collision(self, platforms):
        global counter
        self.can_jump = False
        index = 0
        for platform in platforms:
            if (
                self.rect.left + self.rect.width > platform.rect.left
                and self.rect.left < platform.rect.left + platform.rect.width
                and self.rect.top - self.rect.height >= platform.rect.top
                and self.vert_vel <= 0
            ):
                difference = (self.rect.top - self.rect.height) - platform.rect.top
                if abs(difference) <= abs(self.vert_vel - self.gravity):
                    self.rect.top = platform.rect.top + self.rect.height
                    self.vert_vel = 0
                    self.vert_vel = self.jump_vel

                    if index > self.highest_platform_index:
                        self.highest_platform_index = index
                        counter += 1
                        return "new_platform"
                    return "same_platform"
            index += 1
        
        return "no_platform"
    
    def draw_player(self):
        top = camera.rect.top - self.rect.top
        pygame.draw.rect(win, BLUE, (self.rect.left, top, self.rect.width, self.rect.height))
    
    def inputs(self, move_left, move_right, stay_still):
        if move_left and self.rect.left > 0:
            self.rect.left -= self.player_vel
        if move_right and self.rect.left < screen_width - self.rect.width:
            self.rect.left += self.player_vel

class PlatformManager:
    def __init__(self):
        self.platforms = []
        self.platform_interval = 250
        self.max_platform = 100 + 2 * self.platform_interval 
        

        # you're basically gonna add platforms in here
        self.platforms.append(Platform(random.random() * (screen_width - 100) , 100))
        self.platforms.append(Platform(random.random() * (screen_width - 100), 100 + self.platform_interval))
        self.platforms.append(Platform(random.random() * (screen_width - 100), 100 + self.platform_interval * 2))
    
    def update(self, player: Player):
        self.draw_platforms()
        self.create_platforms(player)

    def create_platforms(self, player):
        # check if the player is above the max
        if player.rect.top > self.max_platform - 2 * self.platform_interval:
            # increase the max_platform
            self.max_platform += self.platform_interval

            # then add new platforms corresponding to the new max
            self.platforms.append(Platform((random.random() * (screen_width - 100)), self.max_platform))
    def remove_off_screen(self):

        index = len(self.platforms) - 1

        while index >= 0:
            off_screen = self.platforms[index].rect.top > screen_height
            if(off_screen):
                # remove the platform
                del self.platforms[index]
            index -= 1
    
    #def add_more_platforms(self, player):
        # check if the player is high enough that it deserves another 
    
    def draw_platforms(self):
        for platform in self.platforms:
            platform.draw()

class Platform():
    def __init__(self, x, y):
        super().__init__()
        self.rect = Box(x, y, 100, 20)
    
    def draw(self):
        top = camera.rect.top - self.rect.top
        pygame.draw.rect(win, BLUE, (self.rect.left, top, self.rect.width, self.rect.height))

def get_current_state():
    platform_index = player.highest_platform_index
    first = (pm.platforms[platform_index].rect.left - player.rect.left) / screen_width 
    second = (pm.platforms[platform_index].rect.top - player.rect.top) / screen_height 
    third = (pm.platforms[platform_index + 1].rect.left - player.rect.left) / screen_width 
    fourth = (pm.platforms[platform_index + 1].rect.top - player.rect.top) / screen_height
    return torch.tensor([first, second, third, fourth])

repeat_bounce_count = 0

def get_current_reward(platform_status, action):
    global repeat_bounce_count
    global player
    global pm
    global memory
    global counter
    global episodes
    reward = 0
    # determine which way it's moving relative to it's platform
    if(action == 1 and player.rect.left < pm.platforms[player.highest_platform_index + 1].rect.left):
        reward += 1
    elif(action == 1 and player.rect.left > pm.platforms[player.highest_platform_index + 1].rect.left):
        reward -= 2
    
    if(action == 0 and player.rect.left > pm.platforms[player.highest_platform_index + 1].rect.left):
        reward += 1
    elif(action == 0 and player.rect.left < pm.platforms[player.highest_platform_index + 1].rect.left):
        reward -= 2
    
    # punish it for staying still for long
    if(action == 2):
        reward -= 0.5
    
    # if new platform has been reached
    if(platform_status == "new_platform"):
        repeat_bounce_count = 0
        reward += 5

        if(counter >= 100):
            episodes.append(player.highest_platform_index)
            player = Player()
            pm = PlatformManager()
            player.rect.left = pm.platforms[0].rect.left
            memory = ReplayMemory(memory.memory.maxlen)
            reward -= 10
            counter = 0
            export_episodes_to_csv(episodes)
    
    # if the player jumped on the same platform
    elif(platform_status == "same_platform"):
        repeat_bounce_count += 1
        if(repeat_bounce_count > 1):
            reward -= 5
    
    # if the player fell off the platform
    if(player.rect.top < pm.platforms[player.highest_platform_index].rect.top):
        episodes.append(player.highest_platform_index)
        player = Player()
        pm = PlatformManager()
        player.rect.left = pm.platforms[0].rect.left
        memory = ReplayMemory(memory.memory.maxlen)
        reward -= 10
        counter = 0
        export_episodes_to_csv(episodes)

    return reward


def perform_memory_update(agent, replay_memory):
    if len(replay_memory.memory) < BATCH_SIZE:
        return

    batch = replay_memory.sample(BATCH_SIZE)

    for sample in batch:
        agent.update_q_values(sample.state, sample.action, sample.next_state, sample.reward, False)
    
    # afterwards, clear the memory
    replay_memory.memory.clear()

    agent.update_target_network()

def export_episodes_to_csv(episodes):
    data = pd.DataFrame()
    data['score'] = episodes

    data.to_csv('export.csv')
    display_graph(episodes)

def display_graph(episodes):
    plt.clf()
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    s = pd.Series(episodes)
    s.plot.line()
    plt.draw()
    plt.pause(0.001)
def plot_bar(q_value, x_position):

    height = q_value * 1250
    height = int(height)
    if(q_value >= 0):
        pygame.draw.rect(win, BLUE, (x_position, screen_height / 2 - height - 100, 100, height + 100))
    else:
        pygame.draw.rect(win, BLUE, (x_position, screen_height / 2, 100, -height))
def plot_q_values(q_values):

    first = q_values[0]
    second = q_values[1]

    if first > second:
        plot_bar(first - second, screen_width)
        plot_bar(0, screen_width + 100)
    else:
        plot_bar(0, screen_width)
        plot_bar(second - first, screen_width + 100)
    # plot_bar(q_values[0], screen_width)
    # plot_bar(q_values[1], screen_width + 100)
    # plot_bar(q_values[2], screen_width + 200)



    # plt.clf()  # Clear the previous plot
    # actions = ['Left', 'Stay', 'Right']
    # plt.bar(actions, q_values)
    # plt.ylim([-2, 2])  # Set the y-axis limits according to your Q-value range
    # plt.xlabel('Actions')
    # plt.ylabel('Q-Values')
    # plt.title('Q-Values')
    # plt.draw()
    # plt.pause(0.001)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# screen configuration
screen_width = 800
screen_height = 600

panel_width = 300

# model configuration
n_observations = 4
n_actions = 3

state_dim = (4)
action_dim = (3)

learning_rate = 0.0005
learning_rate_decay = 0.999999
discount_factor = 0.9
epsilon = 0.9
BATCH_SIZE = 128
MEMORY_SIZE = 2500

# Define colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)



pygame.init()

Font=pygame.font.SysFont('timesnewroman',  30)


win = pygame.display.set_mode((screen_width + panel_width, screen_height))
pygame.display.set_caption("Simple Platform Game")

# initialize the agent
agent = QAgent(state_dim, action_dim, learning_rate, discount_factor, epsilon)
#initialize the replay memory
memory = ReplayMemory(MEMORY_SIZE)
# initialize the camera
camera = Camera(0, screen_height, screen_width, screen_height)
#initialize platform manager
pm = PlatformManager()
# initialize the player
player = Player()
player.rect.left = pm.platforms[0].rect.left

fig, ax = plt.subplots()

font = pygame.font.Font(None, 36) 

episodes = []

# Game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    win.fill(WHITE)

    # collect the current state of the player
    state = get_current_state()
    
    # compute which action to take at the current state
    # returns an integer (0 for left, 1 for right)
    action = agent.select_action(state)

    # put the inputs into the player
    player.inputs(0 == action, 1 == action, 2 == action)
    
    # update the position of the player
    platform_status = player.update()
    pm.update(player)

    # calculate the "next state"
    next_state = get_current_state()

    # calculate the reward
    reward = get_current_reward(platform_status, action)

    # calculate any reward
    memory.push(state, action, next_state, reward)

    # if there are a max number of memories, perform the optimization
    if len(memory.memory) >= memory.memory.maxlen:
        perform_memory_update(agent, memory)
    if counter > high_score:
        high_score = counter

    plot_q_values(q_values1)
    # Blit the counter text onto the window
    counter_text = font.render(f"Counter: {counter}", True, (0, 0, 0))
    max_text = font.render(f"High Score: {high_score}", True, (0, 0, 0))
    epsilon_text = font.render(f"Epsilon: {agent.epsilon}", True, (0, 0, 0))

    counter_rect = counter_text.get_rect()
    counter_rect.topright = (screen_width- 10, 10)
    max_rect = max_text.get_rect()
    max_rect.topright = (screen_width- 10, 30)
    eps_rect = epsilon_text.get_rect()
    eps_rect.topright = (screen_width- 10, 50)

    window.blit(counter_text, counter_rect)
    window.blit(max_text, max_rect)
    window.blit(epsilon_text, eps_rect)
    camera.follow(player)

    # Update the display
    pygame.display.update()

# Quit the game
pygame.quit()
