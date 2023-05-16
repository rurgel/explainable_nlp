import itertools
import nltk
from typing import List

from src.language_processing.text_downloader import WikipediaSummary


nltk.download("punkt", quiet=True)

class TextProcessor(WikipediaSummary):
    """
    A class for processing text.

    Args:
        keyword (str): Term keyword to search into wikipedia.
        language (str, optional): Wikipedia Page Language.
            Defaults to "pt".

    Attributes:
        _text (str): The text to be processed. Inherited from the
            WikipediaTextDownloader class.
        _language_abbrev (str): The abbreviation of the text language.
        paragraph_lengths (List[int]): A list containing the lengths of
            each paragraph in the text.

    Methods:
        split_sentences: Splits the text into a list of sentences.
        model_tokenizer: Tokenizes the sentences using a pre-trained
            language model.
        validate_not_stopwords: Checks if the tokens are not stopwords.
        validate_any_alphanumeric: Checks if the tokens alphanumeric.
        aggregate_tokens: Combines split tokens by ## and also
            aggregates the respective attention weights.
        paragraph_replacer: Replaces the paragraphs in the text with
            their respective aggregated tokens.
        combine_token_attention: Combines the tokens and attention

    Properties:
        language: Returns the full name of the language of the text.
        sentences: Returns a list of sentences in the text.

    """

    _LANGUAGE_DICT: dict = {"pt": "portuguese", "en": "english"}

    def __init__(
        self, keyword: str, *, language: str = "pt", **kwargs
    ) -> None:
        """
        Initialize a TextProcessor object.

        Args:
            keyword (str): Term keyword to search into wikipedia.
            language (str, optional): Wikipedia Page Language.
                Defaults to "pt".

        Raises:
            ValueError: If the input text is empty.
        """
        if not keyword:
            raise ValueError("Keyword cannot be empty")

        self._language_abbrev = language
        # Get text from WikipediaTextDownloader
        super().__init__(keyword, language=language, **kwargs)
        self._split_sentences()

    def _split_sentences(self) -> None:
        """
        Split the text into a list of sentences, and store the list in
        the _sentences attribute.
        """
        paragraphs = self._text.split("\n")
        sentences = [
            nltk.sent_tokenize(paragraph, language=self.language)
            for paragraph in paragraphs
        ]
        self.paragraph_lengths = [len(snt) for snt in sentences if snt]
        self._sentences = list(itertools.chain(*sentences))
        return

    @property
    def sentences(self) -> List[str]:
        """
        Return a list of sentences in the text.

        Returns:
            A list of sentences in the text.
        """
        return self._sentences

    @property
    def keyword(self) -> str:
        return super().keyword

    @keyword.setter
    def keyword(self, kw):
        WikipediaSummary.keyword.fset(self, kw)
        self._split_sentences()

    @property
    def language(self) -> str:
        """
        Return the full name of the language of the text.

        Returns:
            The full name of the language of the text.
        """
        return self._LANGUAGE_DICT.get(self._language_abbrev, "portuguese")


if __name__ == "__main__":
    text = TextProcessor("Python")
    print(text.keyword)
    print(*text.sentences, sep="\n", end="\n\n")

    text.keyword = "Inteligência Artificial"
    print(text.keyword)
    print(*text.sentences, sep="\n")