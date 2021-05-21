from typing import List
from common.sentence import Sentence


class Instance:
    def __init__(self, input: Sentence, output: List[str] = None) -> None:
        """
            Constructor for the instance.
            :param input: sentence containing the words
            :param output: each position has a label list. Because each position has a list of labels.
        """
        self.input = input   # words
        self.output = output  # label lists
        self.id = None  # the numbers
        self.marginals = None
        self.word_ids = None
        self.output_ids = None
        self.is_prediction = None
        self.type = None
        self.content = ""
        self.mentions = None

    def set_id(self, id: int):
        self.id = id

    def set_test_id(self, id: str):
        self.id = id
    def __len__(self) -> int:
        return len(self.input)

