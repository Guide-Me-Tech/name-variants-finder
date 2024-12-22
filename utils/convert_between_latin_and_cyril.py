# ### Janis Rubins - Step 1: Define mappings for character conversions

# Single and multi-character sequences mapping for Latin -> Cyrillic
multi_map_latin_to_cyrillic = {
    "ts": "ц",
    "ch": "ч",
    "sh": "ш",
    "yu": "ю",
    "ya": "я",
    "yo": "ё",
    "zh": "ж",
    "kh": "х",
}

# Individual character mapping for Latin -> Cyrillic
latin_to_russian = {
    "A": "А", "B": "Б", "C": "Ц", "D": "Д", "E": "Е",
    "F": "Ф", "G": "Г", "H": "Х", "I": "И", "J": "Ж",
    "K": "К", "L": "Л", "M": "М", "N": "Н", "O": "О",
    "P": "П", "Q": "К", "R": "Р", "S": "С", "T": "Т",
    "U": "У", "V": "В", "W": "В", "X": "Х", "Y": "Й",
    "Z": "З", "a": "а", "b": "б", "c": "ц", "d": "д",
    "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и",
    "j": "й", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с",
    "t": "т", "u": "у", "v": "в", "w": "в", "x": "x",
    "y": "ы", "z": "з",
}

# Individual character mapping for Cyrillic -> Latin
russian_to_latin = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D",
    "Е": "E", "Ё": "YO", "Ж": "J", "З": "Z", "И": "I",
    "Й": "Y", "К": "K", "Л": "L", "М": "M", "Н": "N",
    "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T",
    "У": "U", "Ф": "F", "Х": "X", "Ц": "S", "Ч": "CH",
    "Ш": "SH", "Щ": "SH", "Ъ": "'", "Ы": "Y", "Ь": "'",
    "Э": "E", "Ю": "YU", "Я": "YA", "а": "a", "б": "b",
    "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "x", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
    "ъ": "'", "ы": "y", "ь": "'", "э": "e", "ю": "yu",
    "я": "ya",
}


# ### Janis Rubins - Step 2: Convert Latin -> Cyrillic with single-pass approach
def latin_to_cyrillic(text):
    """
    Converts text from Latin script to Russian Cyrillic using a combined
    multi-character and single-character mapping.
    """
    # ### Janis Rubins - Step 2.1: Prepare for single-pass scanning
    result = []
    i = 0
    length = len(text)

    while i < length:
        # ### Janis Rubins - Step 2.2: Check if next two chars form a multi-char sequence
        if i + 1 < length:
            two_chars = text[i : i + 2].lower()  # Compare in lowercase for matching
            if two_chars in multi_map_latin_to_cyrillic:
                # Preserve original case if needed, but here we use the mapped Cyrillic
                result.append(multi_map_latin_to_cyrillic[two_chars])
                i += 2
                continue

        # ### Janis Rubins - Step 2.3: Fallback to single char mapping
        char = text[i]
        mapped_char = latin_to_russian.get(char, char)
        result.append(mapped_char)
        i += 1

    return "".join(result)


# ### Janis Rubins - Step 3: Convert Cyrillic -> Latin
def cyrillic_to_latin(text):
    """
    Converts text from Russian Cyrillic to Latin script using a simple
    character mapping.
    """
    # ### Janis Rubins - Step 3.1: Single-pass mapping
    return "".join(russian_to_latin.get(char, char) for char in text)


# ### Janis Rubins - Step 4: Identify script and convert
def identify_and_convert(text):
    """
    Checks if text contains any known Latin or Cyrillic characters;
    returns tuple indicating source/target script plus converted text.
    If neither recognized, returns the original text.
    """
    # ### Janis Rubins - Step 4.1: Check for presence of any known Latin chars
    if any(char in latin_to_russian for char in text):
        return ("uz", "ru", latin_to_cyrillic(text))

    # ### Janis Rubins - Step 4.2: Check for presence of any known Cyrillic chars
    if any(char in russian_to_latin for char in text):
        return ("ru", "uz", cyrillic_to_latin(text))

    # ### Janis Rubins - Step 4.3: Fallback - no known script
    return text
