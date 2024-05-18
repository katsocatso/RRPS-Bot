# Imports
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
import random
import pyspiel
import argparse

MAXRANDOM = 2147483648.0   # 2^31, ratio range is 0 <= r < 1

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
    i = 0
    while not time_step.last():
      i += 1
      # if i >= 5:
      #   break
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
  
  Adapted from markovbails mentioned in "Population-based Evaluation in Repeated Rock-Paper- Scissors as a 
    Benchmark for Multiagent Reinforcement Learning" as well as MrValdez's adaptation in C here
    https://github.com/MrValdez/Roshambo/blob/516dd574403a0e1799c96c83ade3122a3816251f/rsb-iocaine.c#L3477
  """

  def __init__(self, num_actions, name="bot_agent"):
    assert num_actions > 0
    self.num_actions = num_actions  # 3)
    
    markov_length = 243
    
    self.markovindex = 0
    self.nonzeros = 0
    self.window_size = 5
    self.wins = 0
    self.losses = 0
    self.percent_wins = 0.0
    self.percent_losses = 0.0
    
    self.markovchain = np.full((markov_length, 3), 0.33)
    self.markov_use = np.zeros(markov_length)
    self.markov_tally = np.zeros((markov_length, 3))
    
    

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    game, state = pyspiel.deserialize_game_and_state(time_step.observations["serialized_state"])
    
    num_rounds = len(state.history())/2 
    # If we're not in the first round
    if num_rounds > 0:
      # OpenSpiel convention is that we are Player 1 so state.history()
      # is ordered as MyAgentAction1, OpponentAction1, MyAgentAction2, OpponentAction2, etc.
      my_last_action = state.history()[-2]
      opp_last_action = state.history()[-1]
      
      # check if we won or lost the previous round
      # print(f"{state.history()=}")
      # print(f"{state.rewards()=}")
      prev_reward = state.rewards()[0]
      
      if prev_reward > 0:
        self.wins += 1
      elif prev_reward < 0:
        self.losses += 1
      # print(f"{self.wins=}")
      # print(f"{self.losses=}")
      # print(f"{num_rounds=}")
      # print()
      self.percent_wins = self.wins / num_rounds
      self.percent_losses = self.losses / num_rounds
    
    # Update markov chain
     #- element 0 is the number of trials played so far
      #- element i is the action taken on turn i (1 <= i <= trials ) */
    if num_rounds > self.window_size:
        markovindex = 0
        
        for i in range(self.window_size):
          markovindex = (opp_last_action if i == 1 else opp_last_action * 3) 
        
        # Haven't used this row before: zero it, put one in the right place
        # print(self.markov_use)
        if not self.markov_use[markovindex]:
          self.markov_use[markovindex] = 1
          for j in range(3):
            self.markovchain[markovindex][j]=0
          opp_last_action = state.history()[-1]
          self.markovchain[markovindex][opp_last_action] = 1.0 
          self.markov_tally[markovindex][opp_last_action] += 1
        else:
          # We've used it before, so distribute across all used ones
          self.markov_tally[markovindex][opp_last_action] += 1
          
          nonzeros = 0
          
          # count how many have been used (are non-zero)
          for j in range(3):
            nonzeros += self.markov_tally[markovindex][j]
            newprob = 1.0/(nonzeros if nonzeros else 1)
          
          # distribute that value among them
          for j in range(3):
            if self.markov_tally[markovindex][j] > 0:
              self.markovchain[markovindex][j] = newprob * self.markov_tally[markovindex][j]
          # print(f"{self.markovchain=}")
    
    # Decide which action to take
    bail = self.percent_wins >= 0.8 or self.percent_losses >= 0.6
    
    action = random.randint(0,2) 
    # if we're more than 60% behind or ahead, bail (i.e. random pick)
    # if its the first move, also don't use the markov chain

    if num_rounds <= self.window_size or bail:
        # print("bailing")
        action = random.randint(0,2)  
    else:
        # print(self.markovchain)
        # Use markov chain
        markovindex = 0
        
        # traverse backwards
        for i in range(self.window_size)[::-1]:
          markovindex = (opp_last_action if i == 1 else opp_last_action * 3) 


        # generate a continuous uniform variate
        newprob=random.randint(0, MAXRANDOM)/MAXRANDOM
        
        # now do a cumulative probability
        cumprob=0
        for i in range(3):
          cumprob += self.markovchain[markovindex][i]
          if newprob < cumprob:
            action = i % 3
            break
          
        # if(!(newprob<cumprob))  /* test to make sure we don't have floating point error */
        #     retval=0; /*((2+1)%3)*/
    
    # print(f"{action=}")
    probs = np.zeros(self.num_actions)
    probs[action] = 1
    return rl_agent.StepOutput(action=action, probs=probs)


def test_one_bot(bot_id):
  env = rl_environment.Environment(
      "repeated_game(stage_game=matrix_rps(),num_repetitions=" +
      f"{pyspiel.ROSHAMBO_NUM_THROWS}," +
      f"recall={RECALL})",
      include_full_state=True)
  num_players = 2
  num_actions = env.action_spec()["num_actions"]
  
  my_agent = MyAgent(3, name="cat_agent")
  num_wins = 0
  
  print("Evaluating against agent " + roshambo_bot_names[bot_id] + " now")
    
  agents = [
      my_agent,
      create_roshambo_bot_agent(1, num_actions, roshambo_bot_names, bot_id) 
  ]
  avg_eval_returns = eval_agents(env, agents, num_players, 10, verbose=False)
  
  if avg_eval_returns[0] >= 0: # my_agent won
    num_wins += 1 
  
  print("Avg return ", avg_eval_returns)

  return num_wins 

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
  
  # This is to compare my average return against all other bots' average return
  # print(roshambo_bot_cum_avg_return)
  # print("My average return: " + str(my_avg_return))
  return num_wins

# Runs the tests 50 times and returns average number of wins
def full_test_fifty_times():
  total_wins = 0
  
  for i in range(50):
    round_wins = test_myagent()
    total_wins += (round_wins)
  
  return total_wins / 50
  

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple command line program.")
    
    # Add arguments
    parser.add_argument('--testtype', type=str, 
                        help="What mode of testing to use. --fullfifty for all 50 evals, --test for just one.", 
                        required=False)
    parser.add_argument('--botid', type=str,
                       help="Which Roshambo bot to test against.",
                       required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.testtype == "fullfifty":
      print("Num wins over 50 evaluations = " + str(full_test_fifty_times()) + "/43")
    elif args.testtype == "onebot":
      bot_name = roshambo_bot_names[int(args.botid)]
      print(f"Test against bot {bot_name}")
      won = test_one_bot(int(args.botid)) 
      if won:
        print(f"Won against bot {bot_name}!")
      else:
        print(f"Lost against bot {bot_name} :(")
    else:
      print("Num wins = " + str(test_myagent()) + "/43") 
    
# print("Num wins over 50 evaluations = " + str(full_test_fifty_times()) + "/43")
# 23 / 43 = 53.48%
# 24.46 / 43 = 56.88%