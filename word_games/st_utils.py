import numpy as np
import pickle
import copy

rares = "JQXZ"
BLANK = "%"
NO_TILE = " "

# Search
def DFS(i, j, grid, occ, node):
    out = []
    char = grid[i][j]
    occ[i, j] = 0
    pv = []
    if node.is_end_of_word:
        out.append(char)
        pv = [(i, j)]

    # if we're at this node, there must be a word here or below, so
    # setting the best child PV to "at least here" is always correct,
    # even if there may not be a valid word exactly here
    best_child_pv = []
    best_plen = 0
    
    for x in range(max(i-1, 0), min(i+2, len(grid))):
        for y in range(max(j-1, 0), min(j+2, len(grid[0]))):
            # move to x, y
            child = grid[x][y]
            if not occ[x,y] or child not in node.children:
                continue
            
            # legal space and in trie, recurse
            # set occ
            occ[x,y] = 0
            wordlist, child_pv, plen = DFS(x, y, grid, occ, node.children[child])
            if plen > best_plen:
                best_child_pv = child_pv
                best_plen = plen
            out += [char + word for word in wordlist]
            # unset occ after we move out
            occ[x,y] = 1
    occ[i, j] = 1
    # add to pv
    if len(best_child_pv):
        pv = [(i, j)] + best_child_pv
    return out, pv, len(pv)

def follow_pv(grid, pv):
    s = ""
    for (i, j) in pv:
        s += grid[i][j]
    return s

def search(grid, occ, root):
    out = []
    all_pvs = []

    best_pv = []
    best_plen = 0
    longest_word = None
    
    for i in range(len(grid)):
        ilist = []
        for j in range(len(grid[0])):
            if grid[i][j] not in root.children:
                continue
            wordlist, pv, plen = DFS(i, j, grid, occ, root.children[grid[i][j]])
            # filter out 2-letter words
            wordlist = [x for x in wordlist if len(x) > 2]
            plen = 0 if plen < 3 else plen
            
            bestword = follow_pv(grid, pv)
            if plen > best_plen:
                best_plen = plen
                best_pv = pv
                longest_word = bestword
                
            if len(wordlist):
                ilist.append(bestword)
                all_pvs.append(pv)
            else:
                ilist.append(None)
                all_pvs.append(None)
        out.append(ilist)
    return longest_word, best_pv, out, all_pvs

# Game Mechanics
def apply_gravity(grid):
    rows = len(grid)
    cols = len(grid[0])
    is_clear = True

    for col in range(cols):
        # Collect all characters in the current column that are not '%'
        stack = []
        for row in range(rows):
            if grid[row][col] != NO_TILE:
                stack.append(grid[row][col])

        # Fill the column from the bottom with the collected characters
        for row in range(rows - 1, -1, -1):
            if stack:
                is_clear = False # there is still a tile somewhere
                grid[row][col] = stack.pop()
            else:
                grid[row][col] = NO_TILE
    return grid, is_clear

def clear_pv(grid, pv, trie):
    # check that this is a word, for sanity
    word = follow_pv(grid, pv)
    assert trie.is_word(word)
    is_long = len(pv) >= 5
    for (i, j) in pv:
        # if this letter is rare, clear its entire row
        if grid[i][j] in rares:
            for jj in range(len(grid[0])):
                grid[i][jj] = NO_TILE
        grid[i][j] = NO_TILE # clear this letter
        # adjacencies
        for x in range(-1, 2):
            for y in range(-1, 2):
                if (abs(x) == abs(y)) or not((0<= i+x < len(grid)) and (0 <= j+y < len(grid[0]))):
                    continue
                letter = grid[i+x][j+y]
                # long words clear adjacencies, all words clear blanks
                if is_long or letter == BLANK:
                    # don't clear adjacent rares that are in pv so as to not mess up subsequent row clears
                    if (letter not in rares) or ((i+x, j+y) not in pv):
                        grid[i+x][j+y] = NO_TILE
    # now apply gravity
    return word, *apply_gravity(grid)

class SpellTower:
    def __init__(self, grid, trie):
        # make grid into proper format
        grid = [[char for char in row.upper()] for row in grid]
        self.trie = trie
        self.grid = grid
        h, w = len(grid), len(grid[0])
        self.occ= occ = np.ones((h, w))
    
    # just do the longest word, step, repeat
    def greedy_solve(self, grid):
        # grid = copy.deepcopy(grid)
        words = []
        pvs = []
        is_clear = False
        while True:
            longest_word, pv, all_longest, all_pvs = search(grid, self.occ, self.trie.root)
            if longest_word is None:
                break
            _, grid, is_clear = clear_pv(grid, pv, self.trie)
            words.append(longest_word)
            pvs.append(pv)
        return grid, is_clear, words, pvs
    
    def greedy_search(self, word_limit=None):
        # do a BFS that does a greedy rollout at each node until a clear is found
        # tries to find a clear with at most word_limit words
        # not very efficient. Probably can find a solution but not nearly an optimal one
        if word_limit is None:
            word_limit = 1e8
        assert word_limit > 0, "Word Limit must be non-zero and positive"

        best_len = 1e8
        best_pvs = None
        best_words = None
        queue = [(self.grid, [], [])]
        while len(queue):
            subgrid, pv_so_far, words_so_far = queue.pop(0)
            _, _, _, all_pvs = search(subgrid, self.occ, self.trie.root)
            all_pvs = sorted(all_pvs,key=lambda x: (-len(x), x) if x is not None else (1, x))
            for pv in all_pvs:
                if pv is None:
                    continue
                grid = copy.deepcopy(subgrid)
                word, grid, is_clear = clear_pv(grid, pv, self.trie)
                
                # the BFS will eventually consider doing greedy rollouts from this child
                queue.append((copy.deepcopy(grid), pv_so_far + [pv], words_so_far + [word])) 
                _, is_clear, words, pvs = self.greedy_solve(grid)
                if is_clear:
                    new_pvs, new_words = pv_so_far + [pv] + pvs, words_so_far + [word] + words
                    if len(new_words) < best_len:
                        print(f"Found solution of length {len(new_words)}")
                        best_len = len(new_words)
                        best_pvs, best_words = new_pvs, new_words
                    if best_len <= word_limit:
                        return best_words, best_pvs
        print("Could not find any solutions (with only greedy rollouts)")
        return best_words, best_pvs