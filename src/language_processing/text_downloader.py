import re
import wikipedia


class WikipediaSummary:
    def __init__(
        self,
        keyword: str = "Python",
        *,
        language: str = "pt",
        sentences: int = 0,
        clean: bool = True,
    ) -> None:
        """
        Initialize a new WikipediaSummary instance.

        Args:
            keyword (str): The search keyword for fetching the Wikipedia
                summary. Defaults to 'Python'.
            language (str): The language of the Wikipedia to search in.
                Defaults to 'pt' (Portuguese).
            sentences (int): The number of sentences to fetch from the
                summary. A value of 0 fetches the entire summary.
                Defaults to 0.
            clean (bool): Whether to clean the fetched summary by
                removing reference tags and content within parentheses.
                Defaults to True.
        """
        wikipedia.set_lang(language)

        try:
            text = wikipedia.summary(keyword, sentences=0)
        except wikipedia.exceptions.DisambiguationError as e:
            # If the page is ambiguous, choose the first listed page
            keyword = e.options[0]
            text = wikipedia.summary(keyword, sentences=0)
        except wikipedia.exceptions.PageError:
            # If the page does not exist, print an error message
            print(f"Page not found for {keyword} in {language} Wikipedia.")

        self._keyword = keyword
        if clean:
            text = self._remove_references(text)
            text = self._remove_parenthesis(text)
        text = self._fix_numbers(text, language)
        self._text = text

    @property
    def text(self) -> str:
        """Return the text of fetched summary."""
        return self._text

    @staticmethod
    def _remove_references(text: str) -> str:
        """Remove reference tags from the given text."""
        return re.sub(r"\s*\[.*?\]", "", text)

    @staticmethod
    def _remove_parenthesis(text: str) -> str:
        """Remove content within parentheses from the given text."""
        return re.sub(r"\s*\(.*?\)", "", text)

    @staticmethod
    def _fix_numbers(text: str, language: str) -> str:
        """Fix thousand separators in numbers within the given text."""
        thousand_separators = {"pt": ".", "en": ","}
        text = re.sub(r"(\d) (\d)", r"\1\2", text)
        text = re.sub(
            rf"(\d){thousand_separators[language]}(\d)", r"\1\2", text
        )
        return text

    def __repr__(self) -> str:
        """Return a string representation of the instance."""
        repr_str = (
            f"{self._keyword} (from {self.language} Wikipedia):\n"
            f"{self._text}"
        )
        return repr_str

    def __str__(self) -> str:
        """Return the summary as a string."""
        return f"{self._text}"


if __name__ == "__main__":
    ws = WikipediaSummary
    print(ws)
