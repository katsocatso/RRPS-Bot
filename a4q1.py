# Imports
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import regret_matching
import tensorflow.compat.v1 as tf
import pyspiel
import pickle

# Some helper classes and functions.
# DO NOT CHANGE.

class BotAgent(rl_agent.AbstractAgent):
  """Agent class that wraps a bot.

  Note, the environment must include the OpenSpiel state in its observations,
  which means it must have been created with use_full_state=True.

  This is a simple wrapper that lets the RPS bots be interpreted as agents under
  the RL API.
  """

  def __init__(self, num_actions, bot, name="bot_agent"):
    assert num_actions > 0
    self._bot = bot
    self._num_actions = num_actions

  def restart(self):
    self._bot.restart()

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return
    _, state = pyspiel.deserialize_game_and_state(
        time_step.observations["serialized_state"])
    action = self._bot.step(state)
    probs = np.zeros(self._num_actions)
    probs[action] = 1.0
    return rl_agent.StepOutput(action=action, probs=probs)


#  We will use this function to evaluate the agents. Do not change.

def eval_agents(env, agents, num_players, num_episodes, verbose=False):
  """Evaluate the agent.

  Runs a number of episodes and returns the average returns for each agent as
  a numpy array.

  Arguments:
    env: the RL environment,
    agents: a list of agents (size 2),
    num_players: number of players in the game (for RRPS, this is 2),
    num_episodes: number of evaluation episodes to run.
    verbose: whether to print updates after each episode.
  """
  sum_episode_rewards = np.zeros(num_players)
  for ep in range(num_episodes):
    for agent in agents:
      # Bots need to be restarted at the start of the episode.
      if hasattr(agent, "restart"):
        agent.restart()
    time_step = env.reset()
    episode_rewards = np.zeros(num_players)
    while not time_step.last():
    #   print("time_step", time_step)
      agents_output = [
          agent.step(time_step, is_evaluation=True) for agent in agents
      ]
      action_list = [agent_output.action for agent_output in agents_output]
      time_step = env.step(action_list)
      episode_rewards += time_step.rewards
    sum_episode_rewards += episode_rewards
    if verbose:
      print(f"Finished episode {ep}, "
            + f"avg returns: {sum_episode_rewards / (ep+1)}")

  return sum_episode_rewards / num_episodes


def print_roshambo_bot_names_and_ids(roshambo_bot_names):
  print("Roshambo bot population:")
  for i in range(len(roshambo_bot_names)):
    print(f"{i}: {roshambo_bot_names[i]}")

def create_roshambo_bot_agent(player_id, num_actions, bot_names, pop_id):
  name = bot_names[pop_id]
  # Creates an OpenSpiel bot with the default number of throws
  # (pyspiel.ROSHAMBO_NUM_THROWS). To create one for a different number of
  # throws per episode, add the number as the third argument here.
  bot = pyspiel.make_roshambo_bot(player_id, name)
  return BotAgent(num_actions, bot, name=name)


# Some basic info and initialize the population

# print(pyspiel.ROSHAMBO_NUM_BOTS)    # 43 bots
# print(pyspiel.ROSHAMBO_NUM_THROWS)  # 1000 steps per episode

# The recall is how many of the most recent actions are presented to the RL
# agents as part of their observations. Note: this is just for the RL agents
# like DQN etc... every bot has access to the full history.
RECALL = 20

# The population of 43 bots. See the RRPS paper for high-level descriptions of
# what each bot does.

print("Loading bot population...")
pop_size = pyspiel.ROSHAMBO_NUM_BOTS
print(f"Population size: {pop_size}")
roshambo_bot_names = pyspiel.roshambo_bot_names()
roshambo_bot_names.sort()
# print_roshambo_bot_names_and_ids(roshambo_bot_names)

bot_id = 0
roshambo_bot_ids = {}
for name in roshambo_bot_names:
  roshambo_bot_ids[name] = bot_id
  bot_id += 1


# Example: create an RL environment, and two agents from the bot population and
# evaluate these two agents head-to-head.

# Note that the include_full_state variable has to be enabled because the
# BotAgent needs access to the full state.
env = rl_environment.Environment(
    "repeated_game(stage_game=matrix_rps(),num_repetitions=" +
    f"{pyspiel.ROSHAMBO_NUM_THROWS}," +
    f"recall={RECALL})",
    include_full_state=True)
num_players = 2
num_actions = env.action_spec()["num_actions"]
# Learning agents might need this:
# info_state_size = env.observation_spec()["info_state"][0]

# Create two bot agents
p0_pop_id = 0   # actr_lag2_decay
p1_pop_id = 1   # adddriftbot2
agents = [
    create_roshambo_bot_agent(0, num_actions, roshambo_bot_names, p0_pop_id),
    create_roshambo_bot_agent(1, num_actions, roshambo_bot_names, p1_pop_id)
]

# print("Starting eval run.")
# avg_eval_returns = eval_agents(env, agents, num_players, 10, verbose=True)

# print("Avg return ", avg_eval_returns)


####################################################################################################
# My Stuff

#  train agent
def train_agents(env, myAgentIdx, agents, num_players, num_episodes, checkpoint_dir, verbose=False):
  """Evaluate the agent.

  Runs a number of episodes and returns the average returns for each agent as
  a numpy array.

  Arguments:
    env: the RL environment,
    agents: a list of agents (size 2),
    num_players: number of players in the game (for RRPS, this is 2),
    num_episodes: number of evaluation episodes to run.
    verbose: whether to print updates after each episode.
  """
  sum_episode_rewards = np.zeros(num_players)
  for ep in range(num_episodes):
    if ep % 100 == 0:
        agents[myAgentIdx].save(checkpoint_dir)
    for agent in agents:
      # Bots need to be restarted at the start of the episode.
      if hasattr(agent, "restart"):
        agent.restart()
    time_step = env.reset()
    episode_rewards = np.zeros(num_players)
    while not time_step.last():
    #   print(time_step.rewards)
      agents_output = [
          agent.step(time_step, is_evaluation=False) for agent in agents
      ]
    #   print("agents_output", agents_output)
      action_list = [agent_output.action for agent_output in agents_output]
      time_step = env.step(action_list)
      episode_rewards += time_step.rewards
    
    # Episode is over, step all agents with final info state.
    for agent in agents:
        agent.step(time_step)
    
    sum_episode_rewards += episode_rewards
    if verbose:
      print(f"Finished episode {ep}, "
            + f"avg returns: {sum_episode_rewards / (ep+1)}")

  return sum_episode_rewards / num_episodes

#################################################################################################
# Training phase

# Note that the include_full_state variable has to be enabled because the
# BotAgent needs access to the full state.
env = rl_environment.Environment(
    "repeated_game(stage_game=matrix_rps(),num_repetitions=" +
    f"{pyspiel.ROSHAMBO_NUM_THROWS}," +
    f"recall={RECALL})",
    include_full_state=True)
num_players = 2
num_actions = env.action_spec()["num_actions"]
# Learning agents might need this:
info_state_size = env.observation_spec()["info_state"][0]

myAgentIdx = 0

with tf.Session() as sess:
    myAgent = dqn.DQN(
            session=sess,
            player_id=myAgentIdx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=64,
            replay_buffer_capacity=int(1e5),
            batch_size=32)
    
    sess.run(tf.global_variables_initializer())

    # Setting Training Parameters
    checkpoint_dir = './dqn_test'
    num_train_episodes = 500

    print("training using DQN now...")
    
    # # Test against only copybot
    # copybot_id = 7
    # agents = [
    #         myAgent,
    #         create_roshambo_bot_agent(1, num_actions, roshambo_bot_names, copybot_id) 
    #     ]
    # avg_eval_returns = train_agents(env, myAgentIdx, agents, num_players, num_train_episodes, checkpoint_dir, verbose=False)
    # print("Avg return ", avg_eval_returns)
    # myAgent.save(checkpoint_dir) 
    
    for i in range(len(roshambo_bot_names)):
        print("Training against bot " + roshambo_bot_names[i] + " now")
        
        agents = [
            myAgent,
            create_roshambo_bot_agent(1, num_actions, roshambo_bot_names, i) 
        ]
        avg_eval_returns = train_agents(env, myAgentIdx, agents, num_players, num_train_episodes, checkpoint_dir, verbose=False)
        print("Avg return ", avg_eval_returns)
        myAgent.save(checkpoint_dir)


    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    with tf.Session() as sess:
        # Restore variables from disk.
        myAgent.restore(checkpoint_dir)
        model_variables = tf.trainable_variables()
        print("printing saved variables")
        for var in model_variables:
            print(var)
        

# for ep in range(num_train_episodes):
#     time_step = env.reset()
#     while not time_step.last():
#         action = model.step(time_step).action
#         time_step = env.step([action, random.choice(range(num_actions))])  

#     model.step(time_step)  

# model_variables = tf.trainable_variables()
# for var in model_variables:
#     print(var)
# model.save(checkpoint_dir)

# Template : Basic RL agent.
#
#
class MyAgent(rl_agent.AbstractAgent):
  """Agent class that learns to play RRPS.

  You fill this in to create your RRPS agent.

  See the superclass for more info: https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/rl_agent.py
  """

  def __init__(self, num_actions, name="bot_agent"):
    assert num_actions > 0
    self._num_actions = num_actions  # 3
    self.regret_sum = np.zeros(num_actions)
    self.strategy = np.zeros(num_actions)
    self.strategy_sum = np.zeros(num_actions)
    self.tf_session = tf.Session()
    self.agent = dqn.DQN(
            session=sess,
            player_id=myAgentIdx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=64,
            replay_buffer_capacity=int(1e5),
            batch_size=32)
    self.tf_session.run(tf.global_variables_initializer())
    

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return
    # Note: If the environment was created with include_full_state=True, then
    # game and state can be obtained as follows:

    game, state = pyspiel.deserialize_game_and_state(time_step.observations["serialized_state"])
    
    # A useful piece of information is the history (previous actions taken by agents).
    # You can access this by state.history()

    # Do something here that selects an action and computes a policy
    # distribution in probs.
    if len(state.history())==0:
      action = 0
    else:
      action = state.history()[-1]
    probs = np.ones(self._num_actions) / self._num_actions
    return rl_agent.StepOutput(action=action, probs=probs)


# Just trying an example out.

# my_agent = MyAgent(3, name="kate_agent")
# print(my_agent._num_actions)


# p1_pop_id = 1   # adddriftbot2
# agents = [
#     my_agent,
#     create_roshambo_bot_agent(1, num_actions, roshambo_bot_names, p1_pop_id)
# ]


# print("Starting eval run.")
# avg_eval_returns = eval_agents(env, agents, num_players, 10, verbose=True)

# print("Avg return ", avg_eval_returns)