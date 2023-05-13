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
        self._language = language
        self._clean = clean
        self.keyword = keyword

    @property
    def text(self) -> str:
        """Return the text of fetched summary."""
        return self._text

    @property
    def keyword(self) -> str:
        """Return the keyword used to fetch the summary."""
        return self._keyword

    @keyword.setter
    def keyword(self, kw: str) -> None:
        try:
            page = wikipedia.page(kw)
            txt = page.summary
            url = page.url
            title = page.title
        except wikipedia.exceptions.DisambiguationError as e:
            kw = e.options[0]
            page = wikipedia.page(kw)
            txt = page.summary
            url = page.url
            title = page.title
        except wikipedia.exceptions.PageError:
            self._txt = ""
            self._keyword = kw
            raise ValueError(
                f"`{kw}` does not match any pages when searching in "
                "Wikipedia using the selected language."
            )
        self._keyword = kw
        self._url = url
        self._title = title
        if self._clean:
            txt = self._remove_references(txt)
            txt = self._remove_parenthesis(txt)
        txt = self._fix_numbers(txt, self._language)
        self._text = txt
        return

    @property
    def url(self) -> str:
        """Return the url used to fetch the summary."""
        return self._url

    @property
    def title(self) -> str:
        """Return the title used to fetch the summary."""
        return self._title

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
            rf"(\d){re.escape(thousand_separators[language])}(\d)",
            r"\1\2",
            text,
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
    ws = WikipediaSummary("Python")
    print(ws, end="\n\n")

    ws.keyword = "Inteligência Artificial"
    print(ws)
