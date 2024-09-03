import pickle

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def is_word(self, word):
        node = self.root
        return self._is_word(word, node)

    def _is_word(self, word, node):
        return (word[0] in node.children) and \
                ((len(word) == 1 and node.children[word[0]].is_end_of_word) or 
                 (len(word) > 1 and self._is_word(word[1:], node.children[word[0]])))

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

def build_trie_from_file(file_path):
    trie = Trie()
    with open(file_path, 'r') as file:
        for line in file:
            word = line.strip()
            if word:  # Skip empty lines
                trie.insert(word)
    return trie

# rebuild trie
if __name__ == "__main__":
    trie = build_trie_from_file('dictionary.txt')
    trie.save('trie.pkl')