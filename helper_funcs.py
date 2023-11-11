import os
import itertools
import string
from typing import Union, List, Tuple, Optional, BinaryIO, TextIO, ByteString, Dict, Pattern
import regex as re
from colorsys import hsv_to_rgb
import subprocess


def generate_evenly_spaced_colors(n):
    colors = [hsv_to_rgb(i / n, 1, 1) for i in range(n)]
    return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]


def rgb_lumen(color: Tuple[int, int, int]) -> float:
    """
    Calculate the lumen of a color represented as a tuple of RGB values.

    :param color: A tuple of RGB values (each ranging from 0-255).
    :return: A float representing the lumen of the color.
    """

    # Calculate the lumen of the color
    lumen_coeffs = (0.2126, 0.7152, 0.0722)
    return sum([l_coeff * color for l_coeff, color in zip(lumen_coeffs, color)]) / 255


def normalize_luminance(colors: List[Tuple[int, int, int]]) -> list[tuple[int, ...]]:
    """
    Normalize the luminance of a list of color triplets and return them in the order they came.

    :param colors: A list of tuples representing RGB colors (each ranging from 0-255).
    :return: A list of tuples representing RGB colors with normalized luminance.
    """

    # Step 1: Calculate the luminance of each color
    lumens = [rgb_lumen(color) for color in colors]

    # Step 2: Determine the target luminance level (here we take the maximum luminance)
    target_lumen = max(lumens)

    # Step 3 & 4: Calculate the scaling factor and adjust the RGB values
    normalized_colors: List[Tuple[int, ...]] = list()

    for i, color in enumerate(colors):
        if len(color) != 3:
            raise ValueError(f"Invalid color: {color}")

        scaling_factor = target_lumen / lumens[i] if lumens[i] != 0 else 0
        normalized_color = tuple(min(int(channel * scaling_factor), 255) for channel in color)
        normalized_colors.append(normalized_color)

    # Step 5: Return the adjusted colors
    return normalized_colors


# Step 1: Create a function to calculate the distance between two colors
def color_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5


# Step 2: Create a function to find the closest ANSI color to a given RGB color
def rgb_to_ansi(rgb: Tuple[int, int, int]) -> int:
    # Generate a list of all ANSI colors (0-255) and their corresponding RGB values
    ansi_colors = [ansi_to_rgb(i) for i in range(256)]

    # Find the ANSI color with the minimum distance to the input RGB color
    closest_ansi = min(range(256), key=lambda i: color_distance(rgb, ansi_colors[i]))

    return closest_ansi


def ansi_to_rgb(ansi_code: int) -> Tuple[int, int, int]:
    """
    Convert an 8-bit ANSI color code to RGB.

    :param ansi_code: An integer representing an 8-bit ANSI color code (ranging from 0-255).
    :return: A tuple representing the RGB values.
    """
    if not (0 <= ansi_code <= 255):
        raise ValueError("ANSI code must be between 0 and 255")

    if 0 <= ansi_code <= 15:
        # Standard terminal colors (we'll use a predefined mapping)
        standard_colors = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
            (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
            (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
            (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)
        ]
        return standard_colors[ansi_code]
    elif 16 <= ansi_code <= 231:
        # 6x6x6 color cube
        code = ansi_code - 16
        r = (code // 36) * 51
        g = ((code % 36) // 6) * 51
        b = (code % 6) * 51
        return r, g, b
    else:
        # Grayscale
        gray = (ansi_code - 232) * 10 + 8
        return gray, gray, gray


def colored(text: str, fColor: Union[int, Tuple[int, ...]] = (None,), bColor: Union[int, Tuple[int, ...]] = (None,), reset: bool = True, mode_8bit: bool = False) -> str:
    """
    Return a colored string.

    :param text: The text to color.
    :param fColor: Foreground color represented as an ANSI code or a tuple of RGB values (each ranging from 0-255). (optional)
    :param bColor: Background color represented as an ANSI code or a tuple of RGB values (each ranging from 0-255). (optional)
    :param reset: A boolean indicating whether to reset the color at the end of the string. (optional)
    :param mode_8bit: A boolean indicating whether to automatically convert RGB colors to the closest 8-bit ANSI color. (optional)
    :return: A colored string.
    """
    fColorPresent = fColor[0] is not None
    bColorPresent = bColor[0] is not None
    # Convert RGB values to closest ANSI code if the flag is set
    if mode_8bit:
        if fColorPresent and len(fColor) == 3:
            fColor = (rgb_to_ansi(fColor),)
        if bColorPresent and len(bColor) == 3:
            bColor = (rgb_to_ansi(bColor),)

    # Convert ANSI codes to RGB values if necessary
    if fColorPresent and isinstance(fColor[0], int) and len(fColor) == 1:
        fColor = ansi_to_rgb(fColor[0])

    if bColorPresent and isinstance(bColor[0], int) and len(bColor) == 1:
        bColor = ansi_to_rgb(bColor[0])

    # Checking if the colors are valid
    if fColorPresent and len(fColor) not in [1, 3]:
        raise ValueError(f"Invalid foreground color: {fColor}")
    elif bColorPresent and len(bColor) not in [1, 3]:
        raise ValueError(f"Invalid background color: {bColor}")

    # Initializing the reset sequence
    reset = "\033[0m" if reset else ""

    # Building the ANSI code for the background and foreground colors
    if fColorPresent and len(fColor) == 1:
        fColorStr = "\033[38;5;{}m".format(fColor[0])
    elif fColorPresent:
        fColorStr = "\033[38;2;{};{};{}m".format(*fColor)
    else:
        fColorStr = ""

    if bColorPresent and len(bColor) == 1:
        bColorStr = "\033[48;5;{}m".format(bColor[0])
    elif bColorPresent:
        bColorStr = "\033[48;2;{};{};{}m".format(*bColor)
    else:
        bColorStr = ""

    return "".join([bColorStr, fColorStr, text, reset])


def mapVisibleIndex2StrIndex(index: int, patANSI_ESC: Pattern[str], text: str) -> int:
    """
    Map the index of the string with ANSI escape sequences removed to the real index of the string.

    :param index: The index in the string without considering ANSI escape sequences.
    :param patANSI_ESC: A compiled regex pattern to match ANSI escape sequences.
    :param text: The original text string that contains ANSI escape sequences.
    :return: The corresponding index in the text considering ANSI escape sequences.
    """

    # Finding all the ranges of ANSI escape sequences in the text
    esc_ranges = [(match.start(), match.end()) for match in patANSI_ESC.finditer(text)]

    # Initializing the result index and the visible index
    res_index = 0
    visible_index = 0

    # Loop until the visible index reaches the specified index
    while visible_index < index:
        # Traverse through the text; if an ANSI escape sequence is encountered, jump the result index to the end of the sequence
        for esc_range in esc_ranges:
            if esc_range[0] <= res_index <= esc_range[1]:
                res_index = esc_range[1]
                break

        # Increment both the result index and the visible index
        res_index += 1
        visible_index += 1

    # Return the real index considering the ANSI escape sequences
    return res_index


def colored_preserve(text: str, fColor: Tuple[int, ...] = (None,), bColor: Tuple[int, ...] = (None,), colorRange: Tuple[int, int] = (0, -1), mode_8bit: bool = False) -> str:
    """
    Return a colored string while preserving the previous ANSI code modifiers.

    :param text: The text to color.
    :param fColor: Foreground color represented as a tuple of RGB values (each ranging from 0-255).
    :param bColor: Background color represented as a tuple of RGB values (each ranging from 0-255).
    :param colorRange: A tuple representing the range of indices to color. Default is from start to end of the text.
    :return: A colored string.
    """

    # If the color range is default (from start to end), return the colored text without further processing
    if colorRange == (0, -1):
        return colored(text, fColor, bColor, mode_8bit=mode_8bit)

    # Compiling regex patterns for identifying ANSI escape sequences and reset sequences in the text
    ANSI_RST = "\033[0m"
    patANSI_ESC = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    patANSI_RST = re.compile(r'\x1B\[0m')

    # Computing the real range of indices to apply the color, considering ANSI escape sequences
    realRange = (mapVisibleIndex2StrIndex(colorRange[0], patANSI_ESC, text), mapVisibleIndex2StrIndex(colorRange[1], patANSI_ESC, text))

    # Finding the reset sequences within the real range and identifying the closest reset sequence to the end of the real range
    closest_reset = None
    resets = list(patANSI_RST.finditer(text, realRange[1]))
    prevANSI_combinations = []
    prevANSI_string = ""
    if resets:
        closest_reset = max(resets, key=lambda x: x.end())

        # Finding all previous ANSI escape sequences before the closest reset sequence
        prevANSI_combinations = list(patANSI_ESC.finditer(text, 0, realRange[0]))

        # Concatenating all previous ANSI escape sequences into a single string
        prevANSI_string = "".join([text[match.start(): match.end()] for match in prevANSI_combinations])

    # If no previous ANSI combinations are found, return the colored text
    if not prevANSI_combinations:
        return colored(text, fColor, bColor, mode_8bit=mode_8bit)
    # If no closest reset found, color the text and append the previous ANSI strings
    elif not closest_reset:
        return ANSI_RST + colored(text, fColor, bColor, mode_8bit=mode_8bit) + prevANSI_string

    # Creating the colored insert portion by coloring the text within the real range and appending the previous ANSI strings
    colored_insert = colored(text[realRange[0]:realRange[1]], fColor, bColor, mode_8bit=mode_8bit) + prevANSI_string

    # Constructing the final result by inserting the colored portion back into the original text
    result = text[0:realRange[0]] + ANSI_RST + colored_insert + text[realRange[1]:]

    # Returning the final result
    return result


def colour_diff_chars(a: str, b: str) -> str:
    """
    Colour chars from b red, where b ∈ a is not true, where a and b are strings
    :param a:
    :param b:
    :return: coloured string
    """

    result = []  # List to store the resulting characters with coloring
    skipTo = -1  # Index until which characters should be skipped in string 'a'
    bOff = 0  # Offset for indexing string 'b'

    # Loop through the characters of the shorter of the two strings
    for i in range(min(len(a), len(b))):
        # Skip characters in 'a' till index 'skipTo'
        if i < skipTo:
            continue

        ac, bc = a[i], b[i + bOff]  # Current characters from both strings

        # If the characters are the same, append to result and continue
        if ac == bc:
            result.append(bc)
            continue

        seekStart = i + bOff
        # Search for a substring from 'b' in 'a' starting from the current position
        for j in range(seekStart, len(b)):
            offset = j
            try:
                # Check for a substring of length 5 from 'b' in 'a' starting from position 'i'
                if foundAt := a.index(b[offset: offset + 5], i):
                    # Colour the substring in 'b' that's different from 'a' in red
                    redPart = colored(b[i + bOff: foundAt], fColor=(255, 255, 255), bColor=(255, 0, 0))
                    result.append(redPart)

                    # Set the skipTo index to avoid checking already matched substrings
                    skipTo = foundAt

                    # Adjust the offset for string 'b' to align it with the found substring in 'a'.
                    # Here, we calculate the difference between the starting index of the substring in 'b' (offset)
                    # and the position where the substring was found in 'a' (foundAt). This helps in advancing the position
                    # in string 'b' relative to string 'a' for the next iterations
                    bOff = offset - foundAt
                    break
            except ValueError:  # Substring was not found in 'a'
                pass

    return ''.join(result)


def frequency_analysis(message: str, n: int, alphabet: str = string.ascii_lowercase) -> dict:
    # Get up to n-gram frequencies
    frequencies = {i: dict() for i in range(1, n + 1)}
    # Get all permutations of letter pairs of length up to n
    for i in range(1, n + 1):
        for pair in itertools.product(alphabet, repeat=i):
            frequencies[i][''.join(pair)] = 0

    # Count frequencies
    for i in range(len(message) - n + 1):
        for j in range(1, n + 1):
            if message[i:i + j] in frequencies[j]:
                frequencies[j][message[i:i + j]] += 1

    # Change frequencies to percentages of total n-grams
    for i in range(1, n + 1):
        gram_sum = sum(frequencies[i].values())
        for j, gram in enumerate(frequencies[i]):
            frequencies[i][gram] = frequencies[i][gram] / gram_sum * 100

    # Sort frequencies
    for i in range(1, n + 1):
        frequencies[i] = {k: v for k, v in sorted(frequencies[i].items(), key=lambda item: item[1], reverse=True)}

    return frequencies


def print_frequencies(frequencies: dict, top_n: int, decimals: int = 2, digits: int = 6):
    title_len = len(f"Top {top_n} n-grams:" + " " * 4)
    print(*[f"Top {top_n} {i}-grams:" + " " * 4 for i in range(1, len(frequencies) + 1)], sep="\t")
    for i in range(top_n):
        for j in range(1, len(frequencies) + 1):
            gram = sorted(frequencies[j].keys(), key=lambda x: frequencies[j][x], reverse=True)[i:i + 1]
            if not gram:
                print(str.ljust(f"{i + 1:02d}. |", title_len), end="\t")
                continue

            gram = gram[0]
            print(str.ljust(f"{i + 1:02d}. |{gram}: {frequencies[j][gram]: 0{digits}.{decimals}g}%", title_len), end="\t")
        print()


# Put frequencies side by side column wise
def print_frequencies_alt(frequencies: dict, top_n: int, decimals: int = 2, digits: int = 6):
    title_len = len(f"Top {top_n} n-grams:" + " " * 4)
    print(*[f"Top {top_n} {i}-grams:" + " " * 4 for i in range(1, len(frequencies) + 1)], sep="\t")
    for i in range(top_n):
        for j in range(1, len(frequencies) + 1):
            gram = sorted(frequencies[j].keys(), key=lambda x: frequencies[j][x], reverse=True)[i:i + 1]
            if not gram:
                print(str.ljust(f"{i + 1:02d}. |", title_len), end="\t")
                continue

            gram = gram[0]
            print(str.ljust(f"{i + 1:02d}. |{gram}: {frequencies[j][gram]: 0{digits}.{decimals}f}%", title_len), end="\t")
        print()


def openssl_enc(messagePath: str, key: str, enc: str, IV: str = None, decrypt=False) -> subprocess.CompletedProcess:
    enc_dec = ["-e", "-d"]
    out_ext = ["enc", "dec"]

    openssl_enc = {"ECB": "aes-128-ecb",
                   "CBC": "aes-128-cbc",
                   "CFB": "aes-128-cfb",
                   "OFB": "aes-128-ofb"}
    IV_enc = ["CBC", "CFB", "OFB"]

    dirPath = os.path.split(messagePath)[0]
    messageFname = os.path.basename(messagePath).split(".")[0]
    outPath = os.path.join(f"{dirPath}", f"{messageFname}_{enc}.{out_ext[decrypt]}")
    if decrypt:
        outPath = f"{messagePath.split('.')[0]}.{out_ext[decrypt]}"

    cmd = ["openssl", "enc", enc_dec[decrypt], f"-{openssl_enc[enc]}", "-K", key, "-in", messagePath, "-out", outPath]
    if enc in IV_enc:
        cmd.extend(["-iv", IV])

    return subprocess.run(cmd, capture_output=True)


def byte2binaryFormat(byteSeq: ByteString) -> str:
    """
    Convert a byte sequence to a formatted string of binary values.

    :param byteSeq: The byte sequence to convert.
    :return: A formatted string of binary values.
    """
    interRes = [f"{b: 010b}"[-8:].strip() for b in byteSeq]
    for i in range(1, len(interRes) // 4):
        interRes[i * 4 - 1] += "\n"
    return " ".join(interRes)


def byte2hexFormat(byteSeq: ByteString) -> str:
    """
    Convert a byte sequence to a formatted string of hexadecimal values.

    :param byteSeq: The byte sequence to convert.
    :return: A formatted string of hexadecimal values.
    """
    interRes = ["0" + f"{b: 02x}".strip() if len(f"{b: 02x}".strip()) == 1 else f"{b: 02x}".strip() for b in byteSeq]
    for i in range(1, len(interRes) // 4):
        interRes[i * 4 - 1] += "\n"
    return " ".join(interRes)


def colourDifferentBinaryBytes(aDat: str, bDat: str) -> Tuple[str, str]:
    """
    Colour the bytes that differ between two binary strings.

    :param aDat: The first binary string.
    :param bDat: The second binary string.
    :return: The coloured binary strings.
    """
    # High-level pro gamer move
    aListDat = list(aDat)
    bListDat = list(bDat)

    # Color each change byte in red
    skipTo = -1
    indexDiff = 0
    skipTo = -1
    indexDiff = 0
    for i, pair in enumerate(zip(aDat, bDat, strict=True)):
        if skipTo != -1 and i < skipTo:
            continue

        aB, bB = pair
        if aB != bB:
            start = -1
            end = -1
            seek = i
            # Traverse backwards
            while seek >= 0 and aDat[seek].isnumeric():
                seek -= 1
                start = seek

            seek = i
            # Traverse forwards
            while seek < len(aDat) and aDat[seek].isnumeric():
                seek += 1
                end = seek

            splice_start = max(abs(start - indexDiff + 1), 0)
            splice_end = end - indexDiff
            aListDat[splice_start: splice_end] = [colored("".join(aListDat[splice_start: splice_end]), fColor=(255, 0, 0))]
            bListDat[splice_start: splice_end] = [colored("".join(bListDat[splice_start: splice_end]), fColor=(255, 0, 0))]
            indexDiff = len(aDat) - len(aListDat)  # end - start indices were reduced to 1 index
            skipTo = end

    # THIS ISN'T JANK AT ALL
    # I won't take shit from ANY of you
    aDat = "".join(aListDat)
    bDat = "".join(bListDat)
    return aDat, bDat


def colourDifferentHexBytes(aDat: str, bDat: str) -> Tuple[str, str]:
    """
    Colour the bytes that differ between two hex strings.

    :param aDat: The first hex string.
    :param bDat: The second hex string.
    :return: The coloured hex strings.
    """
    # High-level pro gamer move
    aListDat = list(aDat)
    bListDat = list(bDat)

    # Color each change byte in red
    skipTo = -1
    indexDiff = 0
    for i, pair in enumerate(zip(aDat, bDat, strict=True)):
        if skipTo != -1 and i < skipTo:
            continue

        aB, bB = pair
        if aB != bB:
            start = -1
            end = -1
            seek = i
            # Traverse backwards
            while seek >= 0 and aDat[seek].isalnum():
                seek -= 1
                start = seek

            seek = i
            # Traverse forwards
            while seek < len(aDat) and aDat[seek].isalnum():
                seek += 1
                end = seek

            splice_start = max(abs(start - indexDiff + 1), 0)
            splice_end = end - indexDiff
            aListDat[splice_start: splice_end] = [colored("".join(aListDat[splice_start: splice_end]), fColor=(255, 0, 0))]
            bListDat[splice_start: splice_end] = [colored("".join(bListDat[splice_start: splice_end]), fColor=(255, 0, 0))]
            indexDiff = len(aDat) - len(aListDat)  # end - start indices were reduced to 1 index
            skipTo = end

    # THIS ISN'T JANK AT ALL
    # I won't take shit from ANY of you
    aDat = "".join(aListDat)
    bDat = "".join(bListDat)
    return aDat, bDat


def colour_test() -> None:
    msg = "Hello World!"

    # Lumen_TEST section
    test_colors_2 = [
        (100, 100, 100),  # Grey
        (200, 50, 50),  # Darker Red
        (50, 200, 50),  # Darker Green
        (50, 50, 200),  # Darker Blue
        (150, 100, 50),  # Brownish color
    ]

    test_colors_3 = normalize_luminance(test_colors_2)

    print_title("Lumen_TEST")
    print_subtitle("Test PRE")
    print_colors(test_colors_2, msg)

    print_subtitle("Test POST")
    print_colors(test_colors_3, msg)

    # Preserve_TEST section
    bold = "\033[1m"
    underline = "\033[4m"
    italic = "\033[3m"
    reset = "\033[0m"
    pTest_raw = "Bold, Underlined, and ITALICIZED"
    pTest = f"{bold}Bold, {underline}Underlined, and{italic} ITALICIZED{reset}"

    print_title("Preserve_TEST")
    print_subtitle("Test PRE")
    print(pTest)

    print_subtitle("PRESERVE (Cyan)")
    print("Start at 1/3 and end at 2/3")
    print(colored_preserve(pTest, fColor=(0, 255, 255), colorRange=(len(pTest_raw) // 3, len(pTest_raw) // 3 * 2)))

    # Random Colors section
    print_title("Random Colors")
    import math
    lg10 = int(math.log(255 ** 3, 10))
    sg_figs = f"{255 ** 3}"
    sg_figs = int(sg_figs[:2])
    N = input(f"Enter number of colours to generate 0-{sg_figs}E{lg10}:")
    if not N.isnumeric():
        raise ValueError("Invalid number of colours")
    N = int(N)
    print_subtitle("Test PRE")
    test_colors_4 = generate_evenly_spaced_colors(N)
    test_colors_5 = normalize_luminance(test_colors_4)
    print_colors(test_colors_4, msg)

    print_subtitle("Test POST")
    print_colors(test_colors_5, msg)


def print_title(title: str, mode_8bit: bool = False) -> None:
    title_len = len(title)
    print("",
          colored("-" * title_len * 3, fColor=(0, 255, 255), mode_8bit=mode_8bit),
          f"{' ' * title_len}{colored(title, fColor=(255, 0, 255))}{' ' * title_len}",
          colored("-" * title_len * 3, fColor=(0, 255, 255), mode_8bit=mode_8bit),
          sep="\n")


def print_subtitle(subtitle: str, mode_8bit: bool = False) -> None:
    subtitle_len = len(subtitle)
    pad = "-" * (subtitle_len // 2)
    print("",
          f"{pad}{colored('-' * subtitle_len, fColor=(255, 255, 0), mode_8bit=mode_8bit)}{pad}",
          f"{pad.replace('-', ' ')}{subtitle}{' ' * (subtitle_len // 2)}{pad.replace('-', ' ')}",
          f"{pad}{colored('-' * subtitle_len, fColor=(255, 255, 0), mode_8bit=mode_8bit)}{pad}",
          sep="\n")


def print_colors(colors: List[Tuple[int, int, int]], msg: str, mode_8bit: bool = False) -> None:
    for color in colors:
        if len(color) == 3:
            _bit8 = ansi_to_rgb(rgb_to_ansi(color))
        else:
            _bit8 = ansi_to_rgb(color[0])

        print(
            colored(msg, fColor=color, mode_8bit=mode_8bit),
            colored(f"({', '.join(f'{c:02d}' for c in color)})", fColor=color, mode_8bit=mode_8bit),
            "", "",
            colored(msg, fColor=color, mode_8bit=True),
            colored(f"({', '.join(f'{c:02d}' for c in _bit8)})", fColor=color, mode_8bit=True),
            sep="\t"
        )


def print_ansi_palette() -> None:
    ansi_rgb_palette = [ansi_to_rgb(i) for i in range(256)]
    ansi_rgb_palette = normalize_luminance(ansi_rgb_palette)
    for c in range(256):
        if c % 16 == 0:
            print()
        print(f"{colored(f'{c}', fColor=(ansi_to_rgb(c)))}", end="\t")

    for i, nc in enumerate(ansi_rgb_palette):
        if i % 16 == 0:
            print()
        print(f"{colored(f'{i}', fColor=nc)}", end="\t")


def colour_test_8bit() -> None:
    msg = "Hello World!"

    # Lumen_TEST section
    test_colors_2 = [
        (100, 100, 100),  # Grey
        (200, 50, 50),  # Darker Red
        (50, 200, 50),  # Darker Green
        (50, 50, 200),  # Darker Blue
        (150, 100, 50),  # Brownish color
    ]

    test_colors_3 = normalize_luminance(test_colors_2)

    print_title("Lumen_TEST", mode_8bit=True)
    print_subtitle("Test PRE", mode_8bit=True)
    print_colors(test_colors_2, msg, mode_8bit=True)

    print_subtitle("Test POST", mode_8bit=True)
    print_colors(test_colors_3, msg, mode_8bit=True)

    # Preserve_TEST section
    bold = "\033[1m"
    underline = "\033[4m"
    italic = "\033[3m"
    reset = "\033[0m"
    pTest_raw = "Bold, Underlined, and ITALICIZED"
    pTest = f"{bold}Bold, {underline}Underlined, and{italic} ITALICIZED{reset}"

    print_title("Preserve_TEST", mode_8bit=True)
    print_subtitle("Test PRE", mode_8bit=True)
    print(pTest)

    print_subtitle("PRESERVE (Cyan)", mode_8bit=True)
    print("Start at 1/3 and end at 2/3")
    print(colored_preserve(pTest, fColor=(0, 255, 255), colorRange=(len(pTest_raw) // 3, len(pTest_raw) // 3 * 2)))

    # Random Colors section
    print_title("Random Colors", mode_8bit=True)
    import math
    lg10 = int(math.log(255 ** 3, 10))
    sg_figs = f"{255 ** 3}"
    sg_figs = int(sg_figs[:2])
    N = input(f"Enter number of colours to generate 0-{sg_figs}E{lg10}:")
    if not N.isnumeric():
        raise ValueError("Invalid number of colours")
    N = int(N)
    print_subtitle("Test PRE", mode_8bit=True)
    test_colors_4 = generate_evenly_spaced_colors(N)
    test_colors_5 = normalize_luminance(test_colors_4)
    print_colors(test_colors_4, msg, mode_8bit=True)

    print_subtitle("Test POST", mode_8bit=True)
    print_colors(test_colors_5, msg, mode_8bit=True)


if __name__ == "__main__":
    # First 16 ansi 8-bit
    colors1 = [(i,) for i in range(16)]
    print_colors(colors1, "Hello World!")

    colour_test()
    orange = (255, 128, 0)
    orange = normalize_luminance([orange, (150, 150, 150)])[0]

    rtitle = "8-bit ANSI Test"
    title = colored("8-bit ANSI Test", fColor=orange, mode_8bit=True)
    print("-" * 64,
          " " * (64 // 2 - len(rtitle) // 2) + title + " " * (64 // 2 - len(rtitle) // 2),
          "-" * 64,
          sep="\n")

    colour_test_8bit()

    msg1 = "Hello World!"
    msg2 = msg1.replace("Hello", "Hella")
    print(msg1)
    print(colour_diff_chars(msg1, msg2))