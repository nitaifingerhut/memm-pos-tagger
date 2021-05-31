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


PERSONAL_PRONOUNS = (
    "it",
    "you",
    "i",
    "me",
    "myself",
    "he",
    "him",
    "himself",
    "she",
    "herself",
    "we",
    "us",
    "ourselves",
    "who",
    "whom",
    "they",
    "them",
    "themself",
    "themselves",
)


POSSESSIVE_PRONOUNS = ("my", "your", "his", "her", "its", "our", "their")


PUNCTUATION = {
    ".": ".",
    "#": "#",
    "&": "CC",
    ",": ",",
    "%": "NN",
    "``": "``",
    "''": "''",
    "$": "$",
    "--": ":",
    ":": ":",
    "-RRB-": "-RRB-",
}
