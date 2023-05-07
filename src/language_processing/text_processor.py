import itertools
from typing import List, Tuple

import re
import numpy as np
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from src.language_processing.text_downloader import WikipediaSummary


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
        _language_abbrv (str): The abbreviation of the text language.
        _paragraph_lengths (List[int]): A list containing the lengths of
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

    _MODEL_NAME: str = "setu4993/smaller-LaBSE"
    _LANGUAGE_DICT: dict = {"pt": "portuguese", "en": "english"}
    _CONTROL_TOKENS: List[str] = ["[CLS]", "[SEP]"]

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

        # Get text from WikipediaTextDownloader
        super().__init__(keyword, language=language, **kwargs)

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
            tokens = self.tokens(input)
            # Check if no stopwords or alphanumeric in tokens
            att_mask = [
                all(bool_vars)
                for bool_vars in zip(
                    self.validate_not_stopwords(tokens),
                    self.validate_any_alphanumeric(tokens),
                )
            ]
            input["attention_mask"] = torch.tensor(att_mask).unsqueeze(0)

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

    def validate_not_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Check if the tokens are not stopwords.

        Args:
            tokens (List[str]): The list of tokens.

        Returns:
            A list of tokens that are not stopwords.
        """
        stopwords = nltk.corpus.stopwords.words(self.language)
        return [tk.lower() not in stopwords for tk in tokens]

    def validate_any_alphanumeric(self, tokens: List[str]) -> List[str]:
        """
        Check if the tokens contains any alphanumerical character.

        Args:
            tokens (List[str]): The list of tokens.

        Returns:
            A list of tokens that are not stopwords.
        """
        return [bool(re.search(r"\w", tk)) for tk in tokens]

    @property
    def language(self) -> str:
        """
        Return the full name of the language of the text.

        Returns:
            The full name of the language of the text.
        """
        return self._LANGUAGE_DICT.get(self._language_abbrv, "portuguese")

    def aggregate_tokens(
        self, tokens: List[str], attention: torch.Tensor
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Aggregate the attention weights of tokens that were split by
        the tokenizer.

        Args:
            tokens (List[str]): The list of tokens.
            attention (torch.Tensor): The attention weights.

        Returns:
            A tuple containing the list of tokens with the split tokens
            aggregated, and the attention weights with the split tokens
            aggregated.
        """
        for idx in np.where([tk.startswith("##") for tk in tokens])[0][::-1]:
            st_idx = idx - 1

            tokens[st_idx] += tokens[idx].strip("##")
            tokens.pop(idx)
            pos = [x for x in range(attention.shape[-1]) if x != idx]

            # Get the window of the attention tensor
            window = attention[:, :, :, st_idx:idx + 1, st_idx:idx + 1]

            # Find the maximum value along the window 3rd dimension
            max_vals, _ = torch.max(window, dim=3, keepdim=True)

            # Update the window of the attention tensor with max values
            attention[:, :, :, st_idx:idx + 1, st_idx:idx + 1] = max_vals

            # Remove the rows and columns corresponding to split tokens
            attention = attention[:, :, :, pos, :][:, :, :, :, pos]
        return tokens, attention

    @staticmethod
    def reduce_attention(attention: torch.Tensor) -> torch.Tensor:
        """
        Reduce the attention weights to a single value for each
        token.
        Args:
            attention: The attention weights.

        Returns:
            Total attention weight for each token.
        """
        return attention.mean(dim=(0, 1, 2, 3))

    def paragraph_replacer(self, tokens: List[str]) -> List[str]:
        """
        Replace the [SEP] tokens that separate paragraphs with a
        newline character.
        Args:
            tokens: The list of tokens.

        Returns:
            The list of tokens with the [SEP] tokens that separate
        """
        tks = np.array(tokens)
        last_sentences = np.cumsum(self._paragraph_lengths) - 1
        pos = np.where(np.isin(np.where(tks == "[SEP]")[0], last_sentences))[0]
        tks[pos] = "\n"
        return tks.tolist()

    def combine_token_attention(
        self,
        tokens: List[str],
        attention: torch.Tensor,
    ) -> List[Tuple[str, float]]:
        """
        Combine in a zipped list the tokens and the attention weights.
        Control tokens are removed.
        Args:
            tokens: The list of tokens.
            attention: The attention weights.

        Returns:
            A list of tuples containing the tokens and the attention
        """
        return [
            (tk, att.cpu().detach().item())
            for tk, att in zip(tokens, self.reduce_attention(attention))
            if tk not in self._CONTROL_TOKENS
        ]


if __name__ == "__main__":
    # To-Do: Change the main function to unit tests
    text_processor = TextProcessor("Python")

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
