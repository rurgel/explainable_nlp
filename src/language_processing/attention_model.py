from typing import List, Generator, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM
import nltk
import torch
import re
import numpy as np

nltk.download("stopwords", quiet=True)


class AttentionModel:
    """
    A class for computing the attention weights of a text.

    Attributes:
        _tokenizer: A tokenizer object from the transformers library.
        _model: A pre-trained language model from the transformers
            library.

    Methods:
        model_tokenizer: Tokenize the sentences using a pre-trained
            language model.
        attention: Compute the attention weights for each token in the
            input.
        tokens: Convert input IDs to tokens using the tokenizer.
        validate_not_stopwords: Check if the tokens are not stopwords.
        validate_any_alphanumeric: Check if the tokens alphanumeric.
        merge_tokens: Combine split tokens by ## and also
            combine the respective attention weights.
        aggregate_attention: Aggregate the attention weights from the
            paragraph matrix to a single value representing the
            attention weight of the token.
        combine_token_attention: Combine the tokens and attention
            weights into a single list.
        summarize: Summarize the text using the attention weights.

    """

    _MODEL_NAME: str = "setu4993/smaller-LaBSE"
    _REMOVED_TOKENS: List[str] = ["[CLS]", "[SEP]"]
    _END_SENTENCE: str = "[SEP]"

    def __init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_NAME)
        self._model = AutoModelForMaskedLM.from_pretrained(
            self._MODEL_NAME,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=False,
        )

    def model_tokenizer(
        self,
        sentences: Generator[str, None, None],
        language: str,
    ) -> dict:
        """
        Tokenize the sentences using a pre-trained language model.

        Args:
            sentences: A generator of sentences to be tokenized.

        Returns:
            A dictionary containing the input IDs, attention masks,
            and token type IDs.
        """
        inputs = {}
        for sentence in sentences:
            tok_input = self._tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            tokens = self.tokens(tok_input)

            # Check if no stopwords or alphanumeric in tokens
            att_mask = [
                all(bool_vars)
                for bool_vars in zip(
                    self.validate_not_stopwords(tokens, language),
                    self.validate_any_alphanumeric(tokens),
                )
            ]
            tok_input["attention_mask"] = torch.tensor(att_mask).unsqueeze(0)

            for key, value in tok_input.items():
                if key in inputs:
                    inputs[key] = torch.cat((inputs[key], value), dim=1)
                else:
                    inputs[key] = value
        return inputs

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

    @staticmethod
    def validate_not_stopwords(tokens: List[str], language: str) -> List[str]:
        """
        Check if the tokens are not stopwords.

        Args:
            tokens (List[str]): The list of tokens.
            language (str): The language of the text.

        Returns:
            A list of tokens that are not stopwords.
        """
        stopwords = nltk.corpus.stopwords.words(language)
        return [tk.lower() not in stopwords for tk in tokens]

    @staticmethod
    def validate_any_alphanumeric(tokens: List[str]) -> List[str]:
        """
        Check if the tokens contains any alphanumerical character.

        Args:
            tokens (List[str]): The list of tokens.

        Returns:
            A list of tokens that are not stopwords.
        """
        return [bool(re.search(r"\w", tk)) for tk in tokens]

    @staticmethod
    def merge_tokens(
        tokens: List[str], attention: torch.Tensor
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
            window = attention[:, :, :, st_idx : idx + 1, st_idx : idx + 1]

            # Find the maximum value along the window 3rd dimension
            max_vals, _ = torch.max(window, dim=3, keepdim=True)

            # Update the window of the attention tensor with max values
            attention[:, :, :, st_idx : idx + 1, st_idx : idx + 1] = max_vals

            # Remove the rows and columns corresponding to split tokens
            attention = attention[:, :, :, pos, :][:, :, :, :, pos]
        return tokens, attention

    @staticmethod
    def aggregate_attention(attention: torch.Tensor) -> torch.Tensor:
        """
        Aggregate attention matrix to a single value for each token.

        Args:
            attention: The attention weights.

        Returns:
            Total attention weight for each token.
        """
        return attention.mean(dim=(0, 1, 2, 3))

    def combine_token_attention(
        self,
        tokens: List[str],
        attention: torch.Tensor,
        paragraph_lengths: Optional[List[int]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Combine in a zipped list the tokens and the attention weights.
        Control tokens are removed.
        Args:
            tokens: The list of tokens.
            attention: The attention weights.
            paragraph_lengths: The list of paragraph lengths.

        Returns:
            A list of tuples containing the tokens and the attention
        """
        idx = [i for i, s in enumerate(tokens) if s == self._END_SENTENCE]
        attention = self.aggregate_attention(attention)
        attention[idx] = 0
        if paragraph_lengths:
            tokens = self.paragraph_replacer(tokens, paragraph_lengths)
        return [
            (tk, att.cpu().detach().item())
            for tk, att in zip(tokens, attention)
            if tk not in self._REMOVED_TOKENS
        ]

    def paragraph_replacer(
        self, tokens: List[str], paragraph_lengths: List[str]
    ) -> List[str]:
        """
        Replace the [SEP] tokens that separate paragraphs with a
        newline character.
        Args:
            tokens: The list of tokens.
            paragraph_lengths: The list of paragraph lengths.

        Returns:
            The list of tokens with the [SEP] tokens that separate
        """
        tks = np.array(tokens)
        last_sentences = np.cumsum(paragraph_lengths) - 1
        pos = np.where(tks == self._END_SENTENCE)[0][last_sentences]
        tks[pos] = "\n"
        return tks.tolist()

    def summarize(
        self,
        sentences: Generator[str, None, None],
        *,
        language: str,
        paragraph_lengths: Optional[List[int]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get the attention weights for each token in the input.

        Args:
            sentences: A generator of sentences to be tokenized.
            language: The language of the text.
            paragraph_lengths: The list of paragraph lengths.

        Returns:
            A list of tuples containing the tokens and the attention
            weights.
        """
        inputs = self.model_tokenizer(sentences, language)
        attention = self.attention(inputs)
        tokens = self.tokens(inputs)
        tokens, attention = self.merge_tokens(tokens, attention)
        ta = self.combine_token_attention(tokens, attention, paragraph_lengths)
        return ta if ta else [("", 0.0)]


if __name__ == "__main__":
    from src.language_processing.text_processor import TextProcessor

    text = TextProcessor("Inteligência Artificial")
    attention = AttentionModel()
    output = attention.summarize(
        text.sentences,
        language=text.language,
        paragraph_lengths=text.paragraph_lengths,
    )
    print(*output, sep="\n")
