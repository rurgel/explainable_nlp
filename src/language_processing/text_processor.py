import itertools
from typing import List, Tuple

import numpy as np
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class TextProcessor:
    """
    A class for processing text.

    Args:
        text (str): The text to be processed.
        language (str, optional): The language of the text.
            Defaults to "pt".

    Attributes:
        _text (str): The text to be processed.
        _language_abbrv (str): The abbreviation of the text language.
        _language_dict (dict): Dictionary mapping language abbreviations
            to their corresponding full names.
        _paragraph_lengths (List[int]): A list containing the lengths of
            each paragraph in the text.

    Methods:
        split_sentences: Splits the text into a list of sentences.
        model_tokenizer: Tokenizes the sentences using a pre-trained
            language model.
        remove_stopwords: Removes stopwords from a list of tokens.
        aggregate_tokens: Combines split tokens by ## and also
            aggregates the respective attention weights.

    Properties:
        language: Returns the full name of the language of the text.
        sentences: Returns a list of sentences in the text.

    """

    _MODEL_NAME: str = "setu4993/smaller-LaBSE"
    _LANGUAGE_DICT: dict = {"pt": "portuguese", "en": "english"}

    def __init__(self, text: str, *, language: str = "pt") -> None:
        """
        Initialize a TextProcessor object.

        Args:
            text (str): The text to be processed.
            language (str, optional): The language of the text.
                Defaults to "pt".

        Raises:
            ValueError: If the input text is empty.
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        self._text = text
        self._language_abbrv = language
        self._split_sentences()
        self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_NAME)
        self._model = AutoModelForMaskedLM.from_pretrained(
            self._MODEL_NAME,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=False,
        )

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
        self._paragraph_lengths = [len(snt) for snt in sentences if snt]
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

    def model_tokenizer(self) -> dict:
        """
        Tokenize the sentences using a pre-trained language model.

        Returns:
            A dictionary containing the input IDs, attention masks,
            and token type IDs.
        """
        inputs = {}
        for sentence in self.sentences:
            input = self._tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            for key, value in input.items():
                if key in inputs:
                    inputs[key] = torch.cat((inputs[key], value), dim=1)
                else:
                    inputs[key] = value
        return inputs

    def tokens(self, inputs: dict) -> List[str]:
        """
        Convert input IDs to tokens using the tokenizer.

        Args:
            inputs: A dictionary containing the input IDs, attention
                masks, and token type IDs.

        Returns:
            A list of tokens corresponding to the input IDs.
        """
        return self._tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze()
        )

    def attention(self, inputs: dict) -> torch.Tensor:
        """
        Compute the attention weights for each token in the input.

        Args:
            inputs: A dictionary containing the input IDs, attention
                masks, and token type IDs.

        Returns:
            A 5D tensor containing the attention weights with shape
             (num_layers, batch_size, num_layers, tokens, tokens).
        """
        _, _, attentions = self._model(**inputs)
        return torch.stack(attentions)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from a list of tokens.

        Args:
            tokens (List[str]): The list of tokens.

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
        return self._LANGUAGE_DICT.get(self._language_abbrv, "portuguese")

    def aggregate_tokens(
        self, tokens: List[str], attentions: torch.Tensor
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Aggregate the attention weights of tokens that were split by
        the tokenizer.

        Args:
            tokens (List[str]): The list of tokens.
            attentions (torch.Tensor): The attention weights.

        Returns:
            A tuple containing the list of tokens with the split tokens
            aggregated, and the attention weights with the split tokens
            aggregated.
        """
        for idx in np.where([tk.startswith("##") for tk in tokens])[0][::-1]:
            tokens[idx - 1] += tokens[idx].strip("##")
            tokens.pop(idx)
            pos = [x for x in range(attentions.shape[-1]) if x != idx]
            attentions[
                :, :, :, (idx - 1):(idx + 1), (idx - 1):(idx + 1)
            ] = torch.max(
                attentions[
                    :, :, :, (idx - 1):(idx + 1), (idx - 1):(idx + 1)
                ],
                3,
                keepdim=True,
            )[
                0
            ]
            attentions = attentions[:, :, :, pos, :][:, :, :, :, pos]
        return tokens, attentions


if __name__ == "__main__":
    # To-Do: Change the main function to unit tests
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

    paragraph_lengths = text_processor._paragraph_lengths
    for sentence in text_processor.sentences:
        print(sentence)
        paragraph_lengths[0] -= 1
        if paragraph_lengths[0] == 0:
            print("...")
            paragraph_lengths.pop(0)

    inputs = text_processor.model_tokenizer()
    print("Number of tokens:", len(inputs["input_ids"][0]))
    print("Tokenizer keys:", end=" ")
    print(*inputs.keys(), sep=", ")

    tokens = text_processor.tokens(inputs)
    print(tokens[:15])

    attentions = text_processor.attention(inputs)
    print(attentions.shape)

    print(sum([1 for tok in tokens if tok.startswith("##")]))
    tokens, attentions = text_processor.aggregate_tokens(tokens, attentions)
    print(tokens[:15])
    print(len(tokens), attentions.shape)
