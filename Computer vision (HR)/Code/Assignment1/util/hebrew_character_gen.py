import json


class HebrewCharacterGen:
    def __init__(self, path: str) -> None:
        with open(path) as json_file:
            self.name_unicode_dict = json.load(json_file)
        self.int_name_dict = None

    def load_int_name_dict(self, path: str) -> None:
        with open(path) as json_file:
            temp = json.load(json_file)
        self.int_name_dict = {v: k for k, v in temp.items()}

    def give_char(self, character) -> str:
        if type(character) is str:
            return self.name_unicode_dict[character]
        if type(character) is int:
            return self.name_unicode_dict[self.int_name_dict[character]]

    def give_char_name(self, number: int) -> str:
        return self.int_name_dict[number]


if __name__ == "__main__":
    hebrewCharGen = HebrewCharacterGen()
    hebrewCharGen.load_int_name_dict()
    for i in range(27):
        print(f"char {hebrewCharGen.int_name_dict[i]}: {hebrewCharGen.give_char(i)}")
