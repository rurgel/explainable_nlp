import re
import numpy as np
from typing import List, Tuple

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.language_processing.text_processor import TextProcessor
from src.language_processing.attention_model import AttentionModel


class SummaryRenderer:
    _app_historical_keyword: str = "Inteligência Artificial"
    _app_historical_language: str = "pt"


    def __init__(self):
        self._app = FastAPI()
        self._templates = Jinja2Templates(directory="src/app/templates")
        self._app.mount(
            "/static",
            StaticFiles(directory="src/app/static"),
            name="static"
        )

        self._model = AttentionModel()

        @self._app.get("/")
        def index(request: Request):
            return self._templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "keyword": self._app_historical_keyword,
                    "language": self._app_historical_language,
                }
            )

        @self._app.get("/summary")
        async def summary(
            request: Request,
            keyword: str = self._app_historical_keyword,
            language: str = self._app_historical_language,
        ):
            self._app_historical_keyword = keyword
            self._app_historical_language = language
            text = TextProcessor(keyword, language=language)
            tokens_importance = self._model.summarize(
                text.sentences,
                language=text.language,
                paragraph_lengths=text.paragraph_lengths,
            )
            content = self._format_content(tokens_importance)
            return self._templates.TemplateResponse(
                "summary.html",
                {
                    "request": request,
                    "keyword": keyword,
                    "title": text.title,
                    "url": text.url,
                    "language": language,
                    "content": content
                }
            )

        @self._app.exception_handler(ValueError)
        async def user_error_handler(request, exc):
            return self._templates.TemplateResponse(
                "notfound.html",
                {
                    "request": request,
                    "exception": exc,
                    "keyword": self._app_historical_keyword,
                    "language": self._app_historical_language,
                }
            )

    @staticmethod
    def _format_content(
            tokens_importance: Tuple[List[str], List[float]]) -> str:
        tokens, attention = zip(*tokens_importance)
        attention = np.array(attention)

        # Weights by rank to highlight the most important tokens
        attention = attention * np.argsort(np.argsort(attention))
        # Normalize the weights
        attention = (attention - min(attention)) / (
                    max(attention) - min(attention))
        # Scale the weights
        attention = np.round(1.61 ** (2 * attention), 2)
        # Set paragraphs between newlines
        content = [f'<span style="font-size:{att}em"> {tks}</span>'
                   if tks != "\n" else "</p><p>"
                   for tks, att in zip(tokens, attention)]
        # Join the tokens
        content = "".join(content)

        # Remove spaces between punctuation and span tags
        content = re.sub(r"<span[^>]*>\s*([.,:;!?])\s*</span>", r"\1", content)
        # Remove paragraphs at the end of the text
        content = re.sub(r"(</p>)?(<p>)?$", "", content)
        # Remove 1em span tags
        content = re.sub(
            r'<span style="font-size:1\.0em">(.*?)</span>',
            r"\1",
            content)

        return "<p>" + content + "</p>"

    def _start(self):
        import uvicorn
        uvicorn.run(self._app, host="0.0.0.0", port=8080)

    @property
    def app(self):
        return self._app


if __name__ == "__main__":
    SummaryRenderer()._start()

