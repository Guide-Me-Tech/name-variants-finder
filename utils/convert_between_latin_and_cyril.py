# Dictionary for Latin to Russian Cyrillic conversion
latin_to_russian = {
    "A": "А",
    "B": "Б",
    "C": "Ц",
    "D": "Д",
    "E": "Е",
    "F": "Ф",
    "G": "Г",
    "H": "Х",
    "I": "И",
    "J": "Ж",
    "K": "К",
    "L": "Л",
    "M": "М",
    "N": "Н",
    "O": "О",
    "P": "П",
    "Q": "К",
    "R": "Р",
    "S": "С",
    "T": "Т",
    "U": "У",
    "V": "В",
    "W": "В",
    "X": "Х",
    "Y": "Й",
    "Z": "З",
    "a": "а",
    "b": "б",
    "c": "ц",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "x",
    "y": "ы",
    "z": "з",
}
russian_to_latin = {
    "А": "A",
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Д": "D",
    "Е": "E",
    "Ё": "YO",
    "Ж": "J",
    "З": "Z",
    "И": "I",
    "Й": "Y",
    "К": "K",
    "Л": "L",
    "М": "M",
    "Н": "N",
    "О": "O",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "У": "U",
    "Ф": "F",
    "Х": "X",
    "Ц": "S",
    "Ч": "CH",
    "Ш": "SH",
    "Щ": "SH",
    "Ъ": "'",
    "Ы": "Y",
    "Ь": "'",
    "Э": "E",
    "Ю": "YU",
    "Я": "YA",
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sh",
    "ъ": "'",
    "ы": "y",
    "ь": "'",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def latin_to_cyrillic(text):

    text = text.replace("ts", "ц")
    text = text.replace("ch", "ч")
    text = text.replace("sh", "ш")
    text = text.replace("yu", "ю")
    text = text.replace("ya", "я")
    text = text.replace("yo", "ё")
    text = text.replace("zh", "ж")
    text = text.replace("kh", "х")

    result = []
    for char in text:
        result.append(latin_to_russian.get(char, char))
    return "".join(result)


def cyrillic_to_latin(text):

    result = []
    for char in text:
        result.append(russian_to_latin.get(char, char))
    return "".join(result)


def identify_and_convert(text):
    if any([char in latin_to_russian for char in text]):
        return "uz", "ru", latin_to_cyrillic(text)
    if any([char in russian_to_latin for char in text]):
        return "ru", "uz", cyrillic_to_latin(text)
    return text
