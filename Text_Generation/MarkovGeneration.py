import numpy as np
import string
import regex as re


class MarkovTextGenerator:
    def __init__(self, state_size):
        self.state_size = state_size
        self.model = None
        self.corpus = ""

    @staticmethod
    def weighted_random_probabilities(items: list) -> dict:
        total_weight = len(items)
        unique_items, counts = np.unique(items, return_counts=True)
        probabilities = {item: count / total_weight for item, count in zip(unique_items, counts)}
        return probabilities

    @staticmethod
    def inverted_weighted_random_probabilities(items: list) -> dict:
        total_weight = len(items)
        unique_items, counts = np.unique(items, return_counts=True)
        probabilities = {item: count / total_weight for item, count in zip(unique_items, counts)}
        # invert and makesure the sum is 1, use softmax
        soft_max = lambda x: np.exp(x) / np.sum(np.exp(x))
        probabilities = {k: v for k, v in zip(probabilities.keys(), soft_max(list(probabilities.values())))}

        return probabilities

    def load_corpus(self, file_paths, invert=False):
        message = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf8") as f:
                message.append(f.read())
            message.append("\n\n")
        self.corpus = "".join(message)
        self.model = self.build_graph(invert=invert)

    def build_graph(self, invert=False):
        source = self.corpus
        double_space = r"""'"()[]{}<>|_`"""
        single_space = string.punctuation.replace("&", "").replace("/", "").replace("-", "")
        for punc in double_space:
            single_space = single_space.replace(punc, "")

        for punc in single_space:
            source = source.replace(punc, f" {punc}")

        for punc in double_space:
            source = source.replace(punc, f" {punc} ")

        source = source.split()
        graph = {}
        for i in range(self.state_size, len(source)):
            current_word = source[i]
            previous_words = ' '.join(source[i - self.state_size:i])
            graph.setdefault(previous_words, []).append(current_word)

        if invert:
            graph = {k: self.inverted_weighted_random_probabilities(v) for k, v in graph.items()}
            return graph

        graph = {k: self.weighted_random_probabilities(v) for k, v in graph.items()}
        return graph

    def generate_text(self, min_length):
        if not self.model:
            raise ValueError("Model has not been built. Please load a corpus first.")

        def get_new_starter():
            return np.random.choice([k for k in self.model.keys() if k[0].isupper()]).split()

        text = get_new_starter()

        i = self.state_size
        while True:
            key = ' '.join(text[i - self.state_size:i])
            if key not in self.model:
                print(f"Key {key} not in model. Restarting.")
                text.extend(get_new_starter())
                i += 1
                continue

            next_word = np.random.choice(list(self.model[key].keys()), p=list(self.model[key].values()))
            text.append(next_word)
            i += 1
            if i > min_length and text[-1][-1] == '.':
                break

        res = ' '.join(text)
        double_space = r"""'"()[]{}<>|_`"""
        single_space = string.punctuation.replace("&", "").replace("/", "").replace("-", "")
        for punc in double_space:
            single_space = single_space.replace(punc, "")

        for punc in single_space:
            res = res.replace(f" {punc}", punc)

        for punc in double_space:
            res = res.replace(f" {punc} ", punc)

        empty_paren_cit_pat = r"|".join(f"\\{punc}\\s*\\{punc}" for punc in double_space.replace("_", ""))
        empty_paren_cit_pat = re.compile(empty_paren_cit_pat)

        return empty_paren_cit_pat.sub("", res)


# Example usage:
generator = MarkovTextGenerator(state_size=2)
generator.load_corpus(["data/sv_message.txt", "data/sv_message_alt.txt"], invert=True)
generated_text = generator.generate_text(min_length=100)
# Matches: <punctuation><space><capital letter>
_2sentences = re.compile(r"([.?!])\s+([A-Z])")
# \1 is the first group, \2 is the second group
# Replaces with: <punctuation><newline><capital letter>
generated_text = _2sentences.sub(r"\1\n\2", generated_text)
print(generated_text)

print("\n" * 5)

# Example usage:
generator = MarkovTextGenerator(state_size=2)
generator.load_corpus(["data/sv_message.txt", "data/sv_message_alt.txt"], invert=False)
generated_text = generator.generate_text(min_length=100)
# Matches: <punctuation><space><capital letter>
_2sentences = re.compile(r"([.?!])\s+([A-Z])")
# \1 is the first group, \2 is the second group
# Replaces with: <punctuation><newline><capital letter>
generated_text = _2sentences.sub(r"\1\n\2", generated_text)
print(generated_text)