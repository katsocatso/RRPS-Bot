# Roshambo Bot

This repo contains implementation for two bots which play repeated rock paper scissors against a population of 43 other bots, which are described in the paper "Population-based Evaluation in Repeated Rock-Paper- Scissors as a Benchmark for Multiagent Reinforcement Learning."

# Antirotn
The antirotn bot exploits patterns in rotations from the opponent. It can be run simply as `python antirotn.py` to test against the full population.

# Markovbails
The markovbails bot uses a Markov chain model to predict opponent's moves. However, if it is above an 80% win rate or below a 60% loss rate, it will switch to random moves temporarily. 

The file can be run as ```python markovbails.py --testtype=<one of fullfifty, test, onebot> --botid=<id of roshambo bot>```
It accepts two optional command line arguments:
`--testtype` can be one of `fullfifty`, `test`, and `onebot`. 
`test` simply runs it against the full population once. `fullfifty` will evaluate against the full population 50 times and output the average number of wins. `onebot` tests only against one bot specified by the `botid` argument. 

Note: if no arguments are provided, it runs against the full population once (same as `--testtype=test`). 
