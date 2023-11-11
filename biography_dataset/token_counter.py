# This is a work in progress. There are still bugs. Once it is production-ready this will become a full repo.
import os

def count_tokens(text, model_name="gpt-3.5-turbo", debug=False):
    """
    Count the number of tokens in a given text string without using the OpenAI API.
    
    This function tries three methods in the following order:
    1. tiktoken (preferred): Accurate token counting similar to the OpenAI API.
    2. nltk: Token counting using the Natural Language Toolkit library.
    3. split: Simple whitespace-based token counting as a fallback.
    
    Usage:
    ------
    text = "Your text here"
    result = count_tokens(text, model_name="gpt-3.5-turbo", debug=True)
    print(result)
    Required libraries:
    -------------------
    - tiktoken: Install with 'pip install tiktoken'
    - nltk: Install with 'pip install nltk'
    Parameters:
    -----------
    text : str
        The text string for which you want to count tokens.
    model_name : str, optional
        The OpenAI model for which you want to count tokens (default: "gpt-3.5-turbo").
    debug : bool, optional
        Set to True to print error messages (default: False).
    Returns:
    --------
    result : dict
        A dictionary containing the number of tokens and the method used for counting.
    """

    # Try using tiktoken
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(text))
        result = {"n_tokens": num_tokens, "method": "tiktoken"}
        return result
    except Exception as e:
        if debug:
            print(f"Error using tiktoken: {e}")
        pass

    # Try using nltk
    try:
        import nltk
        nltk.download("punkt")
        tokens = nltk.word_tokenize(text)
        result = {"n_tokens": len(tokens), "method": "nltk"}
        return result
    except Exception as e:
        if debug:
            print(f"Error using nltk: {e}")
        pass

    # If nltk and tiktoken fail, use a simple split-based method
    tokens = text.split()
    result = {"n_tokens": len(tokens), "method": "split"}
    return result

class TokenBuffer:
    def __init__(self, max_tokens=2048):
        self.max_tokens = max_tokens
        self.buffer = ""
        self.token_lengths = []
        self.token_count = 0

    def update(self, text, model_name="gpt-3.5-turbo", debug=False):
        new_tokens = count_tokens(text, model_name=model_name, debug=debug)["n_tokens"]
        self.token_count += new_tokens
        self.buffer += text
        self.token_lengths.append(new_tokens)

        while self.token_count > self.max_tokens:
            removed_tokens = self.token_lengths.pop(0)
            self.token_count -= removed_tokens
            self.buffer = self.buffer.split(" ", removed_tokens)[-1]

    def get_buffer(self):
        return self.buffer