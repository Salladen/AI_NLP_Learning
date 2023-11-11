import re
import string

def newline_2_html(text):
    """
    Converts newlines to <br> tags.
    """
    return text.replace("\n", "<br>")