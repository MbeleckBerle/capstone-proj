import requests
from transformers import pipeline
from spellchecker import SpellChecker


class SpellCheck:
    def __init__(self):
        # Initialize the fill-mask pipeline and spell checker
        self.fill_mask = pipeline("fill-mask", model="bert-base-uncased", top_k=100)
        self.spell_checker = SpellChecker()

    @staticmethod
    def get_homophones(word):
        """Fetch homophones of a word using the Datamuse API."""
        response = requests.get(f"https://api.datamuse.com/words?rel_hom={word}")
        return [entry["word"] for entry in response.json()]

    @staticmethod
    def words_only(text):
        """Remove punctuation from the end of words."""
        text_words = text.split()
        text_words_only = []

        for word in text_words:
            if (
                (ord(word[-1]) < 65) or (ord(word[-1]) > 90) and (ord(word[-1]) < 97)
            ) or (ord(word[-1]) > 122):
                word = word[:-1]
            text_words_only.append(word)

        return text_words_only

    def word_check(self, text):
        """Correct spelling using the spell checker and BERT for validation."""
        text = text.lower()
        testing_words = self.words_only(text)
        misspelled = self.spell_checker.unknown(testing_words)

        for word in misspelled:
            candidates = self.spell_checker.candidates(word)

            if len(testing_words) >= 2:
                if word == testing_words[0]:
                    word_sp = word + " "
                    mask = "[MASK] "
                elif word == testing_words[-1]:
                    word_sp = " " + word
                    mask = " [MASK]"
                else:
                    word_sp = " " + word + " "
                    mask = " [MASK] "
            else:
                word_sp = word
                mask = "[MASK]"

            # Replace word with a mask for BERT
            masked_sentence = text.replace(word_sp, mask)

            # Apply BERT to predict the best word
            predictions = self.fill_mask(masked_sentence)

            for prediction in predictions:
                if prediction["token_str"] in candidates:
                    text = masked_sentence.replace("[MASK]", prediction["token_str"])
                    break

        return text.upper()

    def homonym_check(self, text):
        """Check and replace homonyms in the text."""
        text = text.lower()
        testing_words = self.words_only(text)

        for word in testing_words:
            if len(testing_words) >= 2:
                if word == testing_words[0]:
                    word_sp = word + " "
                    mask = "[MASK] "
                elif word == testing_words[-1]:
                    word_sp = " " + word
                    mask = " [MASK]"
                else:
                    word_sp = " " + word + " "
                    mask = " [MASK] "
            else:
                word_sp = word
                mask = "[MASK]"

            # Replace word with a mask for BERT
            masked_sentence = text.replace(word_sp, mask, 1)

            # Apply BERT to predict the best word
            predictions = self.fill_mask(masked_sentence)
            homophones = self.get_homophones(word)

            for prediction in predictions:
                if (prediction["token_str"] == word) or (
                    prediction["token_str"] in homophones
                ):
                    text = masked_sentence.replace("[MASK]", prediction["token_str"])
                    break

        return text.upper()

    def spell_check(self, text):
        """Perform full spell checking and homonym correction."""
        corrected_text = self.word_check(text)
        return self.homonym_check(corrected_text)
