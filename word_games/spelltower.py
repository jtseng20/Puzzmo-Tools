from st_utils import SpellTower
from common.trie import Trie, TrieNode

# just stuff for sanity checking
grid_height = 13
grid_width = 9

# Use % for blank tiles
grid = ["krdtarior",
        "mcdihr%ii",
        "eomohncfe",
        "yoslacajr",
        "ulbeswrsn",
        "necgxhtzt",
        "%iabeiese",
        "flvgygatn",
        "uiuslaepp",
        "dntemiduq",
        "rs%sclo%a",
        "istepsset",
        "ouoneanlp"]

assert len(grid) == grid_height
assert len(set([len(x) for x in grid])) == 1 and len(grid[0]) == grid_width

if __name__ == "__main__":
    loaded_trie = Trie.load('common/trie.pkl')
    solver = SpellTower(grid, loaded_trie)
    
    # greedy solve (repeatedly spell out the longest word possible)
    # _, _, words, coords = solver.greedy_solve(solver.grid)
    
    # greedily search for a solution that clears the board
    words, coords = solver.greedy_search(word_limit=100)
    for word, coord in zip(words, coords):
        print(word, coord)