Running instructions:

We tested this program using python3.12.0 on a unix system in a fresh enviorment.

First install the requirements in REQUIREMENTS.txt, then it should be able to run by just calling

`python main.py "path/to/test/images"`

It should print which scrolls are processed and when it is done (about 2 seconds per scroll). A new directory in the directory from which the programm is executed is created with the name "results". In this directory there are files which have the image name + "_characters.txt" containing the hebrew unicode encoding of these characters. The mapping from char name to unicode can be found in pipeline/data/name_unicode.json