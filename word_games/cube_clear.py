# cube clear is literally the same game as spelltower
from st_utils import SpellTower
from common.trie import Trie, TrieNode

# just stuff for sanity checking
grid_height = 6
grid_width = 6

grid = ["vttide", 
        "oxaldn", 
        "uoauto", 
        "lerccs", 
        "narbqp", 
        "ptrhas"]

assert len(grid) == grid_height
assert len(set([len(x) for x in grid])) == 1 and len(grid[0]) == grid_width

if __name__ == "__main__":
    loaded_trie = Trie.load('common/trie.pkl')
    solver = SpellTower(grid, loaded_trie)
    
    # greedily search for a solution that clears the board
    words, coords = solver.greedy_search(word_limit=3)
    if words is not None:
        for word, coord in zip(words, coords):
            print(word, coord)