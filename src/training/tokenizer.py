import json


UNK_TOKEN = "<unk>"


class CharTokenizer:
    """
    Character-level tokenizer with a fixed vocabulary.

    Unknown characters are mapped to a dedicated <unk> token
    so that encoding never fails on unseen text.
    """

    def __init__(self, chars):

        self.chars = list(chars)

        if UNK_TOKEN not in self.chars:

            self.chars.append(UNK_TOKEN)

        self.char_to_id = {
            char: index
            for index, char in enumerate(self.chars)
        }

        self.id_to_char = {
            index: char
            for index, char in enumerate(self.chars)
        }

        self.unk_id = self.char_to_id[UNK_TOKEN]

    @classmethod
    def build(cls, text):

        unique_chars = sorted(set(text))

        return cls(unique_chars)

    @property
    def vocab_size(self):

        return len(self.chars)

    def encode(self, text):

        ids = [
            self.char_to_id.get(
                char,
                self.unk_id,
            )
            for char in text
        ]

        return ids

    def decode(self, ids):

        return "".join(
            self.id_to_char[int(index)]
            for index in ids
            if int(index) in self.id_to_char
        )

    def save(self, path):

        payload = {
            "chars": self.chars,
        }

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=4,
            )

    @classmethod
    def load(cls, path):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(file)

        return cls(payload["chars"])
