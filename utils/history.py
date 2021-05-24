from typing import Tuple


class History(object):
    def __init__(self, words: Tuple, tags: Tuple, index: int):
        if not all(isinstance(x, str) for x in words):
            raise TypeError
        if not all(isinstance(x, str) for x in tags):
            raise TypeError

        super().__init__()
        self.words = words
        self.tags = tags
        self.index = index

    def __str__(self) -> str:
        words_str = "\n\tWords: " + " , ".join(self.words)
        tags_str = "\n\tTags : " + " , ".join(self.tags)
        index_str = "\n\tIndex: " + str(self.index)
        return f"History:: " + words_str + tags_str + index_str
