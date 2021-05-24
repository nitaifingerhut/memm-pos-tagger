def is_digit(s: str):
    s = s.replace("-", "", 1)
    s = s.replace(".", "", 1)
    s = s.replace(",", "")
    return s.isdigit()


NUMBERS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
)


PUNCTUATION = {".": ".", "#": "#", "&": "CC", ",": ",", "%": "NN", "``": "``", "$": "$"}
