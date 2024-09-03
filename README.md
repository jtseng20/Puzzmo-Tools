# Puzzmo Tools

A few tools for playing the most challenging of the daily puzzles on [Puzzmo](https://www.puzzmo.com/+/sfgate/today).


## Included:
 - Fully vectorized implementation of Pile-Up Poker (Pro), which enables generating ~100k positions per second for efficient RL training
 - Tuned PPO training scripts and pretrained models for Pile-Up Poker and Pile-Up Poker Pro in `pileup_poker/`
 - Solvers for SpellTower, Wordbind, and Cube Clear in `word_games/`


## Pile-Up Poker RL


The two included checkpoints, `pileup_poker/best4x4.pt` and `pileup_poker/best5x5.pt` were both trained over a period of about 12 hours, and play quite solidly (better than me, anyway) but don't perform any search, so their value estimates and decisions very close to the end of games can be a bit off.

| **Model**       | **Average Score / Game** |
|:--------------:|:------------------------:|
| `best4x4.pt`   | ~$5400                   |
| `best5x5.pt`   | ~$9300                   |

### User Input
Users can enter each hand as a space delimited case insensitive string of card names (e.g. `AS`, `JH`, `10D`, `3C`), where the wild card joker in Pile-Up Poker Pro is represented as `WW` .

### Example
```
~/puzzmo_tools/pileup_poker$ python play.py
Play 4x4 or 5x5? [4, 5]: 4
Enter 5 cards for next hand: jc js 9h kd 8s

...

Board:
=====================
| 9H | JC |    | 8S |
=====================
|    |    |    |    |
=====================
|    | JS |    |    |
=====================
|    |    |    |    |
=====================

Hand:
===========================
|    |    |    |    |    |
===========================

Discard:
=====================
| KD |    |    |    |
=====================

...
```
## SpellTower / Cube Clear / Wordbind
The SpellTower / Cube Clear solver runs DFS from every starting point to find all possible words at a given state, then uses length as the search order for a BFS to find a full-clear solution. Game dynamics, e.g. tile clearing and gravity, are factored in end-to-end (except for points / bonus point calculations).

Input grids can be edited in `spelltower.py` and `cube_clear.py`. Inputs are entered at runtime for `wordbind.py`.

All word games use Trie Search on the Scrabble US dictionary, which may or may not exactly match the internal dictionary on Puzzmo.