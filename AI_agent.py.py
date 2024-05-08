# Imports
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
import random
import pyspiel

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


###################################################################################################

# Template : Basic RL agent.
#
#
class MyAgent(rl_agent.AbstractAgent):
  """Agent class that learns to play RRPS.
  
  Adapted from antirotnbot mentioned in "Population-based Evaluation in Repeated Rock-Paper- Scissors as a 
    Benchmark for Multiagent Reinforcement Learning" as well as James Mcguigan's notebook 
    https://www.kaggle.com/code/jamesmcguigan/rock-paper-scissors-anti-rotn-weighted-random/notebook
  """

  def __init__(self, num_actions, name="bot_agent"):
    assert num_actions > 0
    self.num_actions = num_actions  # 3)
    self.rotn_stats = np.zeros(num_actions)   # Keeps track of whether opponent repeated action, went up by one, or two
    self.decay = 0.95   # reduce overfitting
    self.exploration_time = 4   # Play randomly for this number of rounds before switching to antirotn strat

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    game, state = pyspiel.deserialize_game_and_state(time_step.observations["serialized_state"])
    
    # If at least two rounds have been played, update rotn stats
    if len(state.history()) > 4:
      opp_last_action = state.history()[-1]
      opp_second_last_action = state.history()[-3]
      
      # rotn_stat 
      rotn = (opp_last_action - opp_second_last_action) % self.num_actions
      self.rotn_stats[rotn] += 1
      self.rotn_stats *= self.decay
    
    # The first ten rounds, play randomly
    if len(state.history()) <= self.exploration_time:
      action = random.randint(0, self.num_actions - 1)
    else:
      # ExpectedValue(x) = P(x+2) - P(x+1)
      exp_val = np.array([
          (self.rotn_stats[2] - self.rotn_stats[1]),
          (self.rotn_stats[0] - self.rotn_stats[2]),
          (self.rotn_stats[1] - self.rotn_stats[0])    
      ])
      exp_val = exp_val + np.max(exp_val) * 2
      exp_val[exp_val < 0] = 0
      
      # Normalize the expected value
      if sum(exp_val) != 0:
        exp_val = (exp_val / sum(exp_val)).round(3)
        
      offset = random.choices(population=[0,1,2], weights = exp_val, k = 1)[0]
      action = (offset + state.history()[-1]) % self.num_actions
      
    probs = np.zeros(self.num_actions)
    probs[action] = 1
    return rl_agent.StepOutput(action=action, probs=probs)


def test_myagent():
  # Note that the include_full_state variable has to be enabled because the
  # BotAgent needs access to the full state.
  env = rl_environment.Environment(
      "repeated_game(stage_game=matrix_rps(),num_repetitions=" +
      f"{pyspiel.ROSHAMBO_NUM_THROWS}," +
      f"recall={RECALL})",
      include_full_state=True)
  num_players = 2
  num_actions = env.action_spec()["num_actions"]
  
  my_agent = MyAgent(3, name="cat_agent")
  num_wins = 0
  cum_average_return = 0
  
  zeroes = [0] * len(roshambo_bot_names)
  roshambo_bot_cum_avg_return = dict(zip(roshambo_bot_names, zeroes))
  roshambo_bot_avg_return = dict(zip(roshambo_bot_names, zeroes))
  
  for i in range(len(roshambo_bot_names)):
    print("Evaluating against agent " + roshambo_bot_names[i] + " now")
    
    agents = [
        my_agent,
        create_roshambo_bot_agent(1, num_actions, roshambo_bot_names, i) 
    ]
    avg_eval_returns = eval_agents(env, agents, num_players, 10, verbose=False)
    
    if avg_eval_returns[0] >= 0: # my_agent won
      num_wins += 1 
    cum_average_return += avg_eval_returns[0]
    roshambo_bot_cum_avg_return[roshambo_bot_names[i]] += avg_eval_returns[1]
    
    print("Avg return ", avg_eval_returns)
  my_avg_return = cum_average_return / 43
  
  print(roshambo_bot_cum_avg_return)
  print("My average return: " + str(my_avg_return))
  return num_wins

print("Num wins = " + str(test_myagent()) + "/43")