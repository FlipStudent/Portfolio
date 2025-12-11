import random

import docx
import Levenshtein
import pandas as pd
from os import listdir


def docx_to_array(path: str) -> list[list[str]]:
    """
    Reads a docx file to a nested array such that the 2D array represents the original paragraph structure

    :param path: relative or full path to the docx file
    :return: returns a 2D array of characters, where the first dimension represents different lines
    """
    doc = docx.Document(path)
    array = []

    for paragraph in doc.paragraphs:
        line_text = paragraph.text
        line_chars = []
        for char in line_text:
            # This line filters for *empty* and *unicode* characters
            if char == " " or not char.isprintable() or ord(char) in [8207, 1455, 32, 775, 1455]:
                print(char, ord(char))
                continue
            # If we encounter a line break, we want to save our current line and start a new one
            elif char == "\n":
                array.append(line_chars)
                line_chars = []
            # If the character is normal, just append it to the current line
            else:
                line_chars.append(char)
        # If we end the paragraph without a line break and the last line still holds information (characters),
        # forcefully append it to the array
        if len(line_chars) > 0:
            array.append(line_chars)
    return array


# CREDITS TO user Cliff_leaf (Jan 2, 2021)
# https://stackoverflow.com/questions/65537322/python-edit-distance-algorithm-with-dynamic-programming-and-2d-array-output-is
# and https://www.geeksforgeeks.org/edit-distance-dp-5/
def levenshtein_distance(target: list[list[str]], source: list[list[str]]) -> int:
    """
    Calculates the "Levenshtein" or "edit" distance of two nested arrays. 
    Either input variable is an array (paragraph) of arrays (lines) of strings (characters).
    It does not matter which variable is the target and which is the source as the distance is bidirectional

    :param target: the correct array of lines of characters
    :param source: the predicted array of lines of characters
    :return: returns the Levenshtein distance
    """
    t_max_dist = len(target) + 1
    s_max_dist = len(source) + 1
    distance_matrix = [[0 for x in range(s_max_dist)] for x in range(t_max_dist)]

    # Compare letter for letter
    for i in range(t_max_dist):
        for j in range(s_max_dist):
            # Baseline check: if there is nothing to compare yet (0'th letter), 
            # then the distance is simply inserting all letters (=len of other word)
            if i == 0:
                distance_matrix[i][j] = j
            elif j == 0:
                distance_matrix[i][j] = i

                # If the character is correct, the distance remains the same
            elif target[i - 1] == source[j - 1]:
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]
            # Otherwise, at least 1 more step needs to be taken
            # Because your current distance is 1+the shortest distance after your edit,
            # We can use the distance matrix 'sortof' recursively
            else:
                distance_matrix[i][j] = 1 + min(distance_matrix[i][j - 1], distance_matrix[i - 1][j],
                                                distance_matrix[i - 1][j - 1])
    print(distance_matrix)
    # The final outcome after traversing diagonally
    return distance_matrix[t_max_dist - 1][s_max_dist - 1]


def levenshtein_distance_2(target: list[list[str]], source: list[list[str]]) -> int:
    tot_distance = 0

    for idx in range(max(len(target), len(source))):
        if len(target) <= idx:
            string1 = ""
        else:
            string1 = target[idx]

        if len(source) <= idx:
            string2 = ""
        else:
            string2 = source[idx]
        print(f"distance between:\n target:{string1}\n source:{string2} \n is {Levenshtein.distance(string1, string2)}")
        tot_distance += Levenshtein.distance(string1, string2)
    return tot_distance


def read_txt(path: str) -> list[list[str]]:
    file = open(
        path, "r")
    return_string = file.read()
    return_string = return_string.split('\n')
    return_string = filter(None, return_string)
    return_string = [[*y_source_item] for y_source_item in return_string]
    return return_string


def random_source():
    lines = random.randint(12, 35)


if __name__ == "__main__":
    y_source = docx_to_array(f"test_data/124-Fg004_characters.docx")
    # y_source = read_txt(f"results/124-Fg004_characters.txt")
    # print(len(y_source))
    for item in y_source:
        print(len(item))
    exit()

    filenames = ["124-Fg004_characters", "25-Fg001_characters"]
    results = []  # ["settings", "score"]
    for filename in filenames:
        y_target = docx_to_array(f"test_data/{filename}.docx")
        y_source = read_txt(f"results/{filename}.txt")
        print(f"{filename}: levenshtein: {levenshtein_distance_2(y_target, y_source)}")
