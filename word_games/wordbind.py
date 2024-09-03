from common.trie import Trie, TrieNode

def play_wordbind(input_string, trie, min_len=5):
    out = []
    input_string = input_string.upper().strip()
    root = trie.root
    for child in root.children:
        idx = input_string.find(child)
        if idx != -1:
            rest = input_string[idx:]
            wordlist = (helper(rest, root.children[child]))
            out += [x for x in wordlist]
    out = [x for x in out if len(x) > min_len]
    # sanity
    for word in out:
        assert trie.is_word(word), word
    return sorted(out, key=lambda x: -len(x))
    
def helper(input_string, node):
    out = []
    if node.is_end_of_word:
        out.append(input_string[0])
    for child in node.children:
        idx = input_string.find(child)
        if idx != -1:
            wordlist = helper(input_string[idx:], node.children[child])
            out += [input_string[0] + x for x in wordlist]
    return out

if __name__ == "__main__":
    trie = Trie.load('common/trie.pkl')
    words = play_wordbind(input("Enter string of the day (no spaces): "), trie)
    print(words)