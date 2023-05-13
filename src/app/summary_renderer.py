import re
import numpy as np
from typing import List, Tuple
from pydantic import BaseModel

from fastapi import FastAPI, Response, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse

from src.language_processing.text_processor import TextProcessor
from src.language_processing.attention_model import AttentionModel


class SummaryRenderer:
    _app_historical_keyword: str = "Inteligência Artificial"
    _app_historical_language: str = "pt"


    def __init__(self):
        self.app = FastAPI()
        self._templates = Jinja2Templates(directory="src/app/templates")
        self.app.mount(
            "/static",
            StaticFiles(directory="src/app/static"),
            name="static"
        )

        self._model = AttentionModel()

        @self.app.get("/")
        def index(request: Request):
            return self._templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "keyword": self._app_historical_keyword,
                    "language": self._app_historical_language,
                }
            )

        @self.app.get("/summary")
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

        @self.app.exception_handler(ValueError)
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

        @self.app.get("/testme")
        async def testme(
                request: Request,
                keyword: str = self._app_historical_keyword,
                language: str = self._app_historical_language,
        ):
            content = r"""<p><span style="font-size:1.55em"> Python</span> é uma<span style="font-size:1.39em"> linguagem</span> de<span style="font-size:1.45em"> programação</span> de<span style="font-size:1.36em"> alto</span><span style="font-size:1.39em"> nível</span>,<span style="font-size:1.46em"> interpretada</span> de<span style="font-size:1.51em"> script</span>,<span style="font-size:1.12em"> imperativa</span>,<span style="font-size:1.55em"> orientada</span> a<span style="font-size:1.42em"> objetos</span>,<span style="font-size:1.37em"> funcional</span>, de<span style="font-size:1.17em"> tipagem</span><span style="font-size:1.24em"> dinâmica</span> e<span style="font-size:1.26em"> forte</span>. Foi<span style="font-size:1.64em"> lançada</span> por<span style="font-size:1.56em"> Guido</span><span style="font-size:1.31em"> van</span><span style="font-size:1.14em"> Rossum</span> em<span style="font-size:1.78em"> 1991</span>.<span style="font-size:2.03em"> Atualmente</span>,<span style="font-size:1.96em"> possui</span> um<span style="font-size:1.57em"> modelo</span> de<span style="font-size:1.64em"> desenvolvimento</span><span style="font-size:1.19em"> comunitário</span>,<span style="font-size:1.49em"> aberto</span> e<span style="font-size:1.16em"> gerenciado</span> pela<span style="font-size:1.42em"> organização</span> sem<span style="font-size:1.21em"> fins</span><span style="font-size:1.29em"> lucrativos</span><span style="font-size:1.46em"> Python</span><span style="font-size:1.33em"> Software</span><span style="font-size:1.46em"> Foundation</span>.<span style="font-size:2.42em"> Apesar</span> de<span style="font-size:1.62em"> várias</span><span style="font-size:1.32em"> partes</span> da<span style="font-size:1.34em"> linguagem</span><span style="font-size:1.12em"> possuírem</span><span style="font-size:1.28em"> padrões</span> e<span style="font-size:1.43em"> especificações</span><span style="font-size:1.12em"> formais</span>, a<span style="font-size:1.33em"> linguagem</span>, como um<span style="font-size:1.52em"> todo</span>, não é<span style="font-size:1.5em"> formalmente</span><span style="font-size:1.21em"> especificada</span>. O<span style="font-size:1.25em"> padrão</span> na<span style="font-size:1.31em"> pratica</span> é a<span style="font-size:1.54em"> implementação</span><span style="font-size:1.17em"> CPython</span>.</p><p> A<span style="font-size:1.3em"> linguagem</span> foi<span style="font-size:1.16em"> projetada</span> com a<span style="font-size:1.5em"> filosofia</span> de<span style="font-size:1.14em"> enfatizar</span> a<span style="font-size:1.6em"> importância</span> do<span style="font-size:1.39em"> esforço</span> do<span style="font-size:1.23em"> programador</span><span style="font-size:1.32em"> sobre</span> o<span style="font-size:1.31em"> esforço</span><span style="font-size:1.09em"> computacional</span>.<span style="font-size:1.23em"> Prioriza</span> a<span style="font-size:1.16em"> legibilidade</span> do<span style="font-size:1.27em"> código</span><span style="font-size:1.35em"> sobre</span> a<span style="font-size:1.48em"> velocidade</span> ou<span style="font-size:1.13em"> expressividade</span>.<span style="font-size:1.35em"> Combina</span> uma<span style="font-size:1.21em"> sintaxe</span><span style="font-size:1.1em"> concisa</span> e<span style="font-size:1.27em"> clara</span> com os<span style="font-size:1.35em"> recursos</span><span style="font-size:1.41em"> poderosos</span> de sua<span style="font-size:1.29em"> biblioteca</span><span style="font-size:1.27em"> padrão</span> e por<span style="font-size:1.37em"> módulos</span> e<span style="font-size:1.44em"> frameworks</span><span style="font-size:1.61em"> desenvolvidos</span> por<span style="font-size:1.49em"> terceiros</span>.</p><p><span style="font-size:1.4em"> Python</span> é uma<span style="font-size:1.34em"> linguagem</span> de<span style="font-size:1.46em"> propósito</span><span style="font-size:1.24em"> geral</span> de<span style="font-size:1.26em"> alto</span><span style="font-size:1.25em"> nível</span>,<span style="font-size:1.17em"> multiparadigma</span>,<span style="font-size:1.84em"> suporta</span> o<span style="font-size:1.36em"> paradigma</span><span style="font-size:1.39em"> orientado</span> a<span style="font-size:1.31em"> objetos</span>,<span style="font-size:1.1em"> imperativo</span>,<span style="font-size:1.28em"> funcional</span> e<span style="font-size:1.22em"> procedural</span>.<span style="font-size:1.95em"> Possui</span><span style="font-size:1.15em"> tipagem</span><span style="font-size:1.22em"> dinâmica</span> e uma de suas<span style="font-size:1.45em"> principais</span><span style="font-size:1.45em"> características</span> é<span style="font-size:1.77em"> permitir</span> a<span style="font-size:1.49em"> fácil</span><span style="font-size:1.41em"> leitura</span> do<span style="font-size:1.31em"> código</span> e<span style="font-size:1.81em"> exigir</span><span style="font-size:1.57em"> poucas</span><span style="font-size:1.23em"> linhas</span> de<span style="font-size:1.34em"> código</span> se<span style="font-size:1.79em"> comparado</span> ao mesmo<span style="font-size:1.36em"> programa</span> em<span style="font-size:1.72em"> outras</span><span style="font-size:1.15em"> linguagens</span>.<span style="font-size:2.59em"> Devido</span> às suas<span style="font-size:1.52em"> características</span>, ela é<span style="font-size:1.76em"> utilizada</span>,<span style="font-size:1.61em"> principalmente</span>, para<span style="font-size:1.58em"> processamento</span> de<span style="font-size:1.37em"> textos</span>,<span style="font-size:1.32em"> dados</span><span style="font-size:1.43em"> científicos</span> e<span style="font-size:1.56em"> criação</span> de<span style="font-size:1.47em"> CGIs</span> para<span style="font-size:1.44em"> páginas</span><span style="font-size:1.2em"> dinâmicas</span> para a<span style="font-size:1.24em"> web</span>. Foi<span style="font-size:1.68em"> considerada</span> pelo<span style="font-size:1.3em"> público</span> a<span style="font-size:1.19em"> 3ª</span><span style="font-size:1.3em"> linguagem</span> " mais<span style="font-size:1.53em"> amada</span> ", de<span style="font-size:1.4em"> acordo</span> com uma<span style="font-size:1.38em"> pesquisa</span><span style="font-size:1.12em"> conduzida</span> pelo<span style="font-size:1.33em"> site</span><span style="font-size:1.29em"> Stack</span><span style="font-size:1.19em"> Overflow</span> em<span style="font-size:1.77em"> 2018</span> e está entre as<span style="font-size:1.7em"> 5</span><span style="font-size:1.16em"> linguagens</span> mais<span style="font-size:1.63em"> populares</span>, de<span style="font-size:1.42em"> acordo</span> com uma<span style="font-size:1.4em"> pesquisa</span><span style="font-size:1.13em"> conduzida</span> pela<span style="font-size:1.18em"> RedMonk</span>. O<span style="font-size:1.44em"> nome</span><span style="font-size:1.37em"> Python</span> teve a sua<span style="font-size:1.62em"> origem</span> no<span style="font-size:1.41em"> grupo</span><span style="font-size:1.19em"> humorístico</span><span style="font-size:1.3em"> britânico</span><span style="font-size:1.28em"> Monty</span><span style="font-size:1.27em"> Python</span>,<span style="font-size:1.63em"> criador</span> do<span style="font-size:1.38em"> programa</span><span style="font-size:1.25em"> Monty</span><span style="font-size:1.25em"> Python</span> '<span style="font-size:1.21em"> s</span><span style="font-size:1.23em"> Flying</span><span style="font-size:1.29em"> Circus</span>,<span style="font-size:2.17em"> embora</span><span style="font-size:1.59em"> muitas</span><span style="font-size:1.48em"> pessoas</span><span style="font-size:1.51em"> façam</span><span style="font-size:1.35em"> associação</span> com o<span style="font-size:1.11em"> réptil</span> do mesmo<span style="font-size:1.43em"> nome</span>.</p>"""
            return self._templates.TemplateResponse(
                "summary.html",
                {
                    "request": request,
                    "keyword": keyword,
                    "title": "TITLE TEST",
                    "url": "TESTED URL",
                    "language": "en",
                    "content": content
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

    def start(self):
        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    SummaryRenderer().start()

