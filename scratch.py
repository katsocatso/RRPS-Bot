# Imports
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import cfr
import open_spiel
import pyspiel
import pickle

####################################################################################################
# My Stuff
    
#     TurnBasedSimultaneousGame::TurnBasedSimultaneousGame(
#     std::shared_ptr<const Game> game)
#     : Game(ConvertType(game->GetType()),
#            ConvertParams(game->GetType(), game->GetParameters())),
#       game_(game) {}
    
#     std::shared_ptr<const Game> ConvertToTurnBased(const Game& game) {
#   SPIEL_CHECK_EQ(game.GetType().dynamics, GameType::Dynamics::kSimultaneous);
#   return std::shared_ptr<const TurnBasedSimultaneousGame>(
#       new TurnBasedSimultaneousGame(game.shared_from_this()));

def convertType(type):
    type.dynamics = pyspiel.GameType::Dynamics::kSequential
    type.information = GameType::Information::kImperfectInformation
    type.short_name = kGameType.short_name
    type.long_name = "Turn-based " + type.long_name
    type.parameter_specification = kGameType.parameter_specification
    return type

def convertParams(type, params):
    params["name"] = GameParameter(type.short_name)
    GameParameters new_params{{"game", GameParameter{params}}}
    return new_params
    
def convertToTurnBasedSimultaneousGame(game):
    turnBasedGame = pyspiel.Game(convertType(game.type), convertParams(game.params))
    return turnBasedGame
