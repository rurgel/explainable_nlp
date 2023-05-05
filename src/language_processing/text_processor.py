import itertools
from typing import List

import nltk


class TextProcessor:
    """
    A class for processing text.

    Args:
        text (str): The text to be processed.
        language (str, optional): The language of the text. Defaults to "pt".

    Attributes:
        _text (str): The text to be processed.
        _language_abbrv (str): The abbreviation of the language of the text.
        _language_dict (dict): A dictionary mapping language abbreviations
            to their corresponding full names.
        _paragraph_lengths (List[int]): A list containing the lengths of
            each paragraph in the text.

    Methods:
        split_sentences: Splits the text into a list of sentences.
        remove_stopwords: Removes stopwords from a list of tokens.

    Properties:
        language: Returns the full name of the language of the text.

    """

    _language_dict = {"pt": "portuguese", "en": "english"}

    def __init__(self, text: str, *, language: str = "pt") -> None:
        """
        Initialize a TextProcessor object.

        Args:
            text: The text to be processed.
            language: The language of the text.
        """
        self._text = text
        self._language_abbrv = language
        self._paragraph_lengths = []

    def split_sentences(self) -> List[str]:
        """
        Split the text into a list of sentences.

        Returns:
            A list of sentences.
        """
        paragraphs = self._text.split("\n")
        sentences = [
            nltk.sent_tokenize(paragraph, language=self.language)
            for paragraph in paragraphs
        ]
        self._paragraph_lengths = [len(snt) for snt in sentences if snt]

        return list(itertools.chain(*sentences))

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from a list of tokens.

        Args:
            tokens: The list of tokens.

        Returns:
            The list of tokens with stopwords removed.
        """
        stopwords = nltk.corpus.stopwords.words(self.language)
        return [tk for tk in tokens if tk.lower() not in stopwords]

    @property
    def language(self) -> str:
        """
        Return the full name of the language of the text.

        Returns:
            The full name of the language of the text.
        """
        return self._language_dict.get(self._language_abbrv, "portuguese")


if __name__ == "__main__":
    text = (
        "O Brasil é um país de dimensões continentais, com uma área total "
        "de 8.515.767,049 km², sendo o quinto maior país do mundo em área "
        "territorial (equivalente a 47% do território sul-americano) e o sexto em "
        "população (com mais de 210 milhões de habitantes).\nO Brasil é o único "
        "país do mundo que possui fronteiras com 10 outros países. Ele faz "
        "fronteira com todos os outros países sul-americanos, exceto Chile e Equador.\n"
        "A capital do Brasil é Brasília, e a maior cidade é São Paulo. O país é "
        "dividido em 26 estados e um distrito federal."
    )
    text_processor = TextProcessor(text)
    sentences = text_processor.split_sentences()
    paragraph_lengths = text_processor._paragraph_lengths
    for sentence in sentences:
        print(sentence)
        paragraph_lengths[0] -= 1
        if paragraph_lengths[0] == 0:
            print("...")
            paragraph_lengths.pop(0)
