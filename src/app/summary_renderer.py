import re
import numpy as np
from typing import List, Tuple


def format_content(tokens_importance: Tuple[List[str], List[float]]) -> str:
    tokens, attention = zip(*tokens_importance)
    attention = np.array(attention)

    # Weights by rank to highlight the most important tokens
    attention = attention * np.argsort(np.argsort(attention))
    # Normalize the weights
    attention = (attention-min(attention))/(max(attention)-min(attention))
    # Scale the weights
    attention = np.round(1.61**(2*attention), 2)
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

    return "<p>"+content+"</p>"


if __name__ == "__main__":
    import uvicorn

    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse

    from src.language_processing.text_processor import TextProcessor
    from src.language_processing.attention_model import AttentionModel

    model = AttentionModel()
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def root():
        text = TextProcessor("Python")
        tokens_importance = model.summarize(
            text.sentences,
            language=text.language,
            paragraph_lengths=text.paragraph_lengths,
        )
        content = format_content(tokens_importance)
        body = f"<body><h1>{text.keyword}</h1>{content}</body>"
        html_content = f"<html>{body}</html>"
        return Response(content=html_content, media_type="text/html")
    uvicorn.run(app, host="0.0.0.0", port=8000)