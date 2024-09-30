import langdetect
import functools
import nltk
import re
import collections
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from utils import read_examples
from scipy.special import expit, softmax
import numpy as np
topic_label = {
    'arts_&_culture':0, 'business_&_entrepreneurs':1, 'celebrity_&_pop culture':2, 'daily life':3, 'family':4, 
    'fashion_&_style':5, 'film_&_tv_&_video':6, 'fitness_&_health':7, 'food_&_dining':8, 'gaming':9,
    'learning_&_educational':10, 'music':11, 'social concern':12, 'other_hobbies':13, 'relationships':14,
    'science_&_technology':15, 'sports':16, 'travel_&_adventure':17, 'youth_&_student life':18
}
label_topic = {value: key for key, value in topic_label.items()}


label_sentiment1 = {
    0:'negative sentiment', 1:'neutral sentiment', 2:'positive sentiment'
}
label_sentiment2 = {
    0:'anger', 1:'disgust', 2:'fear', 3:'joy', 4:'neutral sentiment', 5:'sadness', 6:'surprise'
}





class topic_classifier:

    def __init__(self,topic):
        # load models
        MODEL = f"cardiffnlp/tweet-topic-21-multi"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to("cuda")
        self.topic  = topic
        #model.to(args.device)
        return
    
    def feed_back(self,value):
        predicted_topic, is_correct = self.classify_topic(value)
        if is_correct == 1:
            return True
        else:
            return (f"The detected topic of the response is '{predicted_topic}', which does not match the expected topic '{self.topic}'.\n"
                    f"Please adjust the content to align more closely with the topic '{self.topic}'.")
        

    def classify_topic(self, text):
        label = self.topic
        '''
        refer to https://huggingface.co/cardiffnlp/tweet-topic-21-multi
        '''
        tokens = self.tokenizer(text, return_tensors='pt').to("cuda")
        if tokens.input_ids.shape[1] > 512:
            tokens.input_ids = tokens.input_ids[:, :512]
            tokens.attention_mask = tokens.attention_mask[:, :512]

        output = self.model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
        output = output[0][0].detach().cpu()
        scores = output.numpy()
        scores = expit(scores)
        pred = np.argmax(scores)
        predictions = (scores >= 0.3) * 1
        
        return label_topic[pred], predictions[topic_label[label]]



class sentiment_classifier:
    def __init__(self,sentiment):
        self.sentiment = sentiment
        # load models 1
        MODEL1 = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer1 = AutoTokenizer.from_pretrained(MODEL1)
        self.model1 = AutoModelForSequenceClassification.from_pretrained(MODEL1)
        self.model1.to("cuda")
        # load models 2
        MODEL2 = f"j-hartmann/emotion-english-roberta-large"
        self.tokenizer2 = AutoTokenizer.from_pretrained(MODEL2)
        self.model2 = AutoModelForSequenceClassification.from_pretrained(MODEL2)
        self.model2.to("cuda")

    def feed_back(self,value):

        predicted_sentiment, correct1 = self.classify_sentiment(value)
        if correct1==1:
            return True
        else:
            return (f"The sentiment of the text is '{predicted_sentiment}', which does not match the required sentiment '{self.sentiment}'.\n"
                    f"Please adjust the sentiment of the text to be more '{self.sentiment}'.")

        
    

    def classify_sentiment(self, text):
        label = self.sentiment
        '''
        refer to https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
        '''
        model, tokenizer = None, None
        if label in ['negative sentiment', 'neutral sentiment', 'positive sentiment']:
            model = self.model1
            tokenizer = self.tokenizer1
        elif label in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
            model = self.model2
            tokenizer = self.tokenizer2

        encoded_input = tokenizer(text, return_tensors='pt').to("cuda")
        if encoded_input.input_ids.shape[1] > 512:
            encoded_input.input_ids = encoded_input.input_ids[:, :512]
            encoded_input.attention_mask = encoded_input.attention_mask[:, :512]

        output = model(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
        output = output[0][0].detach().cpu()
        scores = output.numpy()
        scores = softmax(scores)
        pred = np.argmax(scores)
        if len(scores) == 3:
            return label_sentiment1[pred], label_sentiment1[pred] == label
        elif len(scores) == 7:
            return label_sentiment2[pred], label_sentiment2[pred] == label










class response_language:
    # Dictionary mapping language codes to full names
    
    def __init__(self, language):
        self._language = language
        
    # here, we want to return a string telling LLM that the response is not the right language.
    def feed_back(self, value):
        self.detect_language(value)

        if (self.detected_l == None) or (self.detected_l == self._language):
            return True
        else:
            error_feedback = self.error_message()
            return error_feedback


    def detect_language(self, value):

        try:
            self.detected_l = langdetect.detect(value)

        except langdetect.LangDetectException as e:
            self.detected_l = None

    
    
    def error_message(self):
        
        """return the language of the given text.

        Args:
        value: A string representing the response.

        Returns:
        True if the language of `value` follows instruction; otherwise False.
        """
        #assert isinstance(value, str)
        self.LANGUAGE_MAP = {
        'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 'ca': 'Catalan', 
        'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 
        'es': 'Spanish', 'et': 'Estonian', 'fa': 'Persian', 'fi': 'Finnish', 'fr': 'French', 'gu': 'Gujarati', 
        'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'id': 'Indonesian', 'it': 'Italian', 
        'ja': 'Japanese', 'kn': 'Kannada', 'ko': 'Korean', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mk': 'Macedonian', 
        'ml': 'Malayalam', 'mr': 'Marathi', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'pa': 'Punjabi', 
        'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak', 'sl': 'Slovenian', 
        'so': 'Somali', 'sq': 'Albanian', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 
        'th': 'Thai', 'tl': 'Tagalog', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese', 
        'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)', 'zh': 'Chinese'
        }

        #langdetect.detect(value) == self._language
        
        detected_language = self.LANGUAGE_MAP.get(self.detected_l, 'Unknown Language')
        target_language = self.LANGUAGE_MAP.get(self._language, 'Unknown Language')
        #if detected_language == 'Unknown Language':
        #    detected_language 
        error_message = (
            f"The response language is '{detected_language}', "
            f"which does not match the required language '{target_language}'."
        )

        return error_message


@functools.lru_cache(maxsize=None)
def _get_sentence_tokenizer():
    return nltk.data.load("nltk:tokenizers/punkt/english.pickle")

def count_sentences(text):
    """Count the number of sentences."""
    tokenizer = _get_sentence_tokenizer()
    tokenized_sentences = tokenizer.tokenize(text)
    return len(tokenized_sentences)



# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")



class NumberOfSentences:

    def __init__(self, num_sentences=None, relation=None):
        self._num_sentences_threshold = num_sentences
        self._comparison_relation = relation
    
#    def count_sentences(self,value):
#        num_sentences = count_sentences(value)
  
    # 
    def feed_back(self, value):
        num_sentences = count_sentences(value)
        #print(num_sentences)
        #exit()
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            if num_sentences < self._num_sentences_threshold:
                return True
            else:
                error_message = self.error_message(num_sentences)
                return error_message

        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            if num_sentences >= self._num_sentences_threshold:
                return True
            else:
                error_message = self.error_message(num_sentences)
                return error_message
        
    
    def error_message(self,num_sentences):
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            
            sentences_to_remove = num_sentences - self._num_sentences_threshold + 1
            error_msg = (f"The response contains {num_sentences} sentences, which is {sentences_to_remove} more than allowed.\n"
                        f"Please remove at least {sentences_to_remove} sentences.")
        else:
            sentences_to_add = self._num_sentences_threshold - num_sentences
            error_msg = (f"The response contains only {num_sentences} sentences.\n"
                             f"Please add at least {sentences_to_add} more sentences to meet the minimum required of {self._num_sentences_threshold} sentences.")
        return error_msg


class Placeholder:
    def __init__(self,num_placeholders):
        self._num_placeholders = num_placeholders
        
    def feed_back(self, value):
        """Check if the number of placeholders follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual number of placeholders in the response is greater than
          or equal to `num_placeholders`; otherwise, False.
        """
        placeholders = re.findall(r"\[.*?\]", value)
        #print(placeholders)
        num_placeholders = len(placeholders)
        
        
        if num_placeholders >= self._num_placeholders:
            return True
        else:
            return self.error_message(placeholders)

    def error_message(self,placeholders):
        num_placeholders = len(placeholders)
        if len(placeholders) != 0:
            # Join the placeholders into a string, separated by commas
            placeholders_list = ", ".join(placeholders)
            return (
                f"The response contains {num_placeholders} placeholders: {placeholders_list}.\nPlease, add at least {self._num_placeholders-num_placeholders} more placeholder(s) such as [restaurant]."
            )
        else:
            return f"The response contains {num_placeholders} placeholders.\nPlease, add at least {self._num_placeholders} more placeholder(s) such as [restaurant]."




class BulletList:
    def __init__(self,num_bullets):
        self._num_bullets = num_bullets
    

    def feed_back(self, value):
        r"""Check if the number of bullet lists meets the requirement.

        Args:
          value: A string representing the response. The response is expected to
            contain some bullet lists that start with `\*`.

        Returns:
          True if the actual number of bullet lists in the response meets the
          requirement.
        """
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)


        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        
        if num_bullet_lists == self._num_bullets:
            return True
        else: 
            return self.error_message(num_bullet_lists,bullet_lists,bullet_lists_2)

    def error_message(self,num_bullet_lists,bullet_lists,bullet_lists_2):
        # Combine both types of bullet points into a single list
        all_bullet_lists = bullet_lists + bullet_lists_2
    
        # Truncate each bullet point if it's too long (e.g., more than 40 characters)
        truncated_bullets = []
        if len(all_bullet_lists) !=0:
            
                
            for bullet in all_bullet_lists:
                bullet = bullet.strip()
                if len(bullet) > 30:
                    truncated_bullets.append(bullet[:40] + "...")
                else:
                    truncated_bullets.append(bullet)
    
            # Create a summary of the bullets
            bullet_summary = "\n".join(truncated_bullets)
            # Construct the final error message
            if len(all_bullet_lists) > self._num_bullets:
                add_num =len(all_bullet_lists)-self._num_bullets
                error_message = (
                    f"In the response, there are {num_bullet_lists} bullet points.\n"
                    f"Here are the bullet points detected:\n{bullet_summary}\n"
                    f"Please remove exactly {add_num} bullet points to meet the requirement of {self._num_bullets}."
                )
            else:
                shortage = self._num_bullets - num_bullet_lists
                error_message = (
                    f"In the response, there are {num_bullet_lists} bullet points.\n"
                    f"Here are the bullet points detected:\n{bullet_summary}\n"
                    f"This is {shortage} fewer than needed. Please add exactly {shortage} more bullet points to meet the requirement of {self._num_bullets}."
                )
        else:
            error_message = (
                f"In the response, there is no bullet points.\n"
                f"We need exactly {self._num_bullets} number of bullet points."
            )
        return error_message


class EndChecker:
    """Checks that the prompt ends with a given phrase."""

    def __init__(self,end_phrase):
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        

    def feed_back(self, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        
        if value.endswith(self._end_phrase):
            return True
        else: 
            return self.error_message(value)

    def error_message(self,value):
        
        return f"The response ends with: '{value[-30:]}'.\nPlease make sure it ends exactly with '{self._end_phrase}'"


class Postscript:

    def __init__(self,postscript_marker):
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )

    def feed_back(self, value):
        """Checks if the response follows the postscript format.

        Args:
          value: a string representing the response. The response is expected to
            contain a postscript section.

        Returns:
          True if the response contains a postscript section starting with
          the keyword containing in the `instruction_args`; otherwise False.
        """
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        if postscript:
            return True
        else:
            return (f"The response does not contain the postscript: {self._postscript_marker}\n"
                    f"Please, add the postscript starting with: {self._postscript_marker}")


class KeywordChecker:
    def __init__(self,keywords):
        self._keywords = keywords
    
    def feed_back(self, value):
        """Check if the response contain the expected keywords."""
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return f"The reponse does not have the keyword {keyword}.\nPlease add the keyword {keyword}." 
        return True


class Excludekeyword:
    def __init__(self,forbidden_words):
        self._forbidden_words = list(set(forbidden_words))
    
    def feed_back(self, value):
        """Check if the response does not contain the expected keywords."""
        for word in self._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return f"The reponse contains the forbidden word: {word}. Please remove it."  
        return True 

class LetterFrequency:

    def __init__(self,letter=None, let_frequency=None, let_relation=None):
        self._letter = letter.strip()
        self._letter = self._letter.lower()
        self._frequency = let_frequency
        self._comparison_relation = let_relation
    def feed_back(self, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            if letters[self._letter] < self._frequency:
                return True
            else:
                return self.error_message(letters[self._letter])
        else:
            if letters[self._letter] >= self._frequency:
                return True
            else:
                return self.error_message(letters[self._letter])
        
    def error_message(self, actual_frequency):

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            excess = actual_frequency - self._frequency + 1
            message = f"The response contains the letter '{self._letter}' {actual_frequency} times, which is {excess} too many.\nPlease remove {excess} occurrences to have less than {self._frequency}."
        else:
            shortage = self._frequency - actual_frequency
            message = f"The response contains the letter '{self._letter}' {actual_frequency} times, which is {shortage} too few.\nPlease add {shortage} more occurrences to meet at least {self._frequency}."

        return message




class KeywordFrequencyChecker:
    def __init__(self,keyword=None, frequency=None, relation=None):
        self._keyword = keyword.strip()
        self._frequency = frequency
        self._comparison_relation = relation
    
    def feed_back(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            if actual_occurrences < self._frequency:
                return True
            else:
                return self.error_message(actual_occurrences)
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            if actual_occurrences >= self._frequency:
                return True
            else:
                return self.error_message(actual_occurrences)

    def error_message(self, actual_occurrences):
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            excess = actual_occurrences - self._frequency + 1
            message = (
                f"The response contains the keyword '{self._keyword}' {actual_occurrences} times, which is {excess} too many."
                f"\nPlease remove {excess} keyword '{self._keyword}' to make sure it occurs less than {self._frequency}."
            )
        else: 
            if actual_occurrences > 0:
                shortage = self._frequency - actual_occurrences
                message = (
                    f"The response contains the keyword '{self._keyword}' {actual_occurrences} times, which is {shortage} too few."
                    f"\nPlease add {shortage} more '{self._keyword}' to meet at least {self._frequency} instances."
                )
            else:
                message = (
                    f"The response does not contain the keyword '{self._keyword}'."
                    f"\nAt least {self._frequency} '{self._keyword}' are required. Please add {self._frequency} instances."
                )

        return message


def count_words(text):
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    num_words = len(tokens)
    return num_words

class NumberOfWords:
    def __init__(self,relation,num_words):
        self._num_words = num_words
        self._comparison_relation = relation
    
    def feed_back(self, value):
        """Checks if the response contains the expected number of words."""
        num_words = count_words(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            if num_words < self._num_words:
                return True
            else:
                return self.error_message(num_words)
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            if num_words >= self._num_words:
                return True
            else:
                return self.error_message(num_words)

    def error_message(self, actual_num_words):
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            excess = actual_num_words - self._num_words +1
            message = (
                f"The response contains {actual_num_words} words, which is {excess} too many. "
                f"\nPlease remove {excess} words to have fewer than {self._num_words} words."
            )
        else: 
            shortage = self._num_words - actual_num_words
            message = (
                f"The response contains only {actual_num_words} words, which is {shortage} too few. "
                f"\nPlease add {shortage} more words to meet at least {self._num_words} words."
            )
        return message


# the instruction asks llm to output a text separated by *** into #num_paragraphs# of sections

class ParagraphChecker:
    def __init__(self,num_paragraphs):
        self._num_paragraphs = num_paragraphs

    def feed_back(self, value):

        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
          if not paragraph.strip():
            if index == 0 or index == len(paragraphs) - 1:
                num_paragraphs -= 1
            else:

                return  f"The response has empty section between the separators'***'.\nPlease, ensure we have exactly {self._num_paragraphs} sections with content and no empty space between two '***'."
        #print(num_paragraphs)
        
        if num_paragraphs == self._num_paragraphs:
            return True
        else:
            return self.error_message(num_paragraphs)

    def error_message(self,num_paragraphs):
        if num_paragraphs < self._num_paragraphs:
            difference = self._num_paragraphs - num_paragraphs
            if difference == 1:
                return f"Expected exactly {self._num_paragraphs} sections, but found only {num_paragraphs}.\nPlease add 1 more section by adding 1 more '***'."
            else:
                return f"Expected exactly {self._num_paragraphs} sections, but found only {num_paragraphs}.\nPlease add {difference} more sections by adding {difference} more '***'."
        else:
            difference = num_paragraphs - self._num_paragraphs
            if difference == 1:
                return f"Expected exactly {self._num_paragraphs} sections, but found {num_paragraphs}.\nPlease remove 1 section by removing 1 '***'."
            else:
                return f"Expected exactly {self._num_paragraphs} sections, but found {num_paragraphs}.\nPlease remove {difference} sections by removing {difference} '***'." 
        

_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.", "My answer is no.", "My answer is maybe.")


class ConstrainedResponseChecker:
    def __init__(self):
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS

    def feed_back(self, value):

        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return "The response doesn't match any of the expected answers.\nPlease respond with exactly one of the following: 'My answer is yes.', 'My answer is no.', 'My answer is maybe.'"






class HighlightSectionChecker:
    def __init__(self,num_highlights):
        self._num_highlights = num_highlights
    
    def feed_back(self, value):
        """Checks if the number of highlighted sections meets the requirement.

        Args:
            value: a string repesenting the response. The response is expected to
            contain highlighted sections in the format of *highlighted*.

        Returns:
        True if the actual number of highlighted sections in the format of
        *highlighed sections* meets the minimum requirement; otherwise False.
        """
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1

        if num_highlights >= self._num_highlights:
            return True
        else:
            return self.error_message(num_highlights)
    
    def error_message(self,num_highlights):
        difference = self._num_highlights - num_highlights
        if difference == 1:
            return f"The response contains {num_highlights} highlighted section.\nPlease include 1 more highlighted section in the format of *highlighted*."
        else:
            return f"The response contains {num_highlights} highlighted sections.\nPlease include at least {difference} more highlighted sections in the format of *highlighted*."

import json

class JsonFormat:
    def __init__(self):
        pass
    def feed_back(self, value):
        value = (
        value.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
        )
        try:
            json.loads(value)
        except ValueError as _:
            return "The response is not in Json format.\nPlease reformat the whole response in Json format."
        return True




class TitleChecker:
    def __init__(self):
        pass
    def feed_back(self, value):
        """Checks if the response contains a title."""
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)

        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return "The response does not contain any title.\nPlease add a title like this: <<title>>."

class QuotationChecker:
    def __init__(self):
        pass
    def feed_back(self, value):
        """Checks if the response is wrapped with double quotation marks."""
        value = value.strip()
        if len(value) > 1 and value[0] == '"' and value[-1] == '"':
            return True
        else:
            return "The whole response is not wrapped with double quotation marks.\nPlease wrap the response in double quotation marks: \"your response\"."


class CommaChecker:
    def __init__(self):
        pass 
    def feed_back(self, value):
        _context_size = 10
        """Checks that the response does not contain commas."""
        comma_positions = [(m.start(), m.end()) for m in re.finditer(r",", value)]
        if not comma_positions:
            return True
        else:
            contexts = []
            for pos in comma_positions:
                start = max(0, pos[0] - _context_size)
                end = min(len(value), pos[1] + _context_size)
                context = value[start:end].strip()
                contexts.append(f"({context})")
            num_comma = len(comma_positions)
            return (
                f"The response contains {num_comma} comma(s). "
                f"Here are the detected commas: {' '.join(contexts)}.\n"
                "Please remove all commas."
            )


class CapitalallLetters:

    def __init__(self):
        pass
    
    def feed_back(self, value):
        """Checks that the response is in English and in all capital letters."""
        assert isinstance(value, str)

        # Split the input string into words
        words = value.split()
        
        # Find words that are not fully in uppercase
        lower_case_words = [word for word in words if any(char.islower() for char in word)]

        if value.isupper():
            return True
        else:
            return f"The response contains words that are not in all capital letters: {', '.join(lower_case_words)}.\nPlease capitalize all of them."

class CapitalWordFrequencyChecker:
    def __init__(self,capital_frequency,capital_relation):
        self._frequency = capital_frequency
        self._comparison_relation = capital_relation
    
    def feed_back(self, value):
        """Checks the frequency of words with all capital letters."""
        # Hyphenated words will count as one word
        words = nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words = len(capital_words)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            if capital_words < self._frequency:
                return True
            else:
                return self.error_message(capital_words)
        else:
            if capital_words >= self._frequency:
                return True
            else:
                return self.error_message(capital_words)
    def error_message(self,num_capital):
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return f"The response contains {num_capital} capitalized words, which is more than the allowed {self._frequency-1}.\nPlease remove at least {num_capital - self._frequency+1} capitalized word(s)."
        else:
            return f"The response contains {num_capital} capitalized words, which is less than the required {self._frequency}.\nPlease add at least {self._frequency - num_capital} more capitalized word(s)."
        

class LowercaseLetter:
    def __init__(self):
        pass
    def feed_back(self, value):

        # Split the input string into words
        words = value.split()
        
        # Find words that are not fully in lowercase
        upper_case_words = [word for word in words if any(char.isupper() for char in word)]

        if value.islower():
            return True
        else:
            return f"The response contains words that are not in all lowercase letters: {', '.join(upper_case_words)}.\nPlease lowercase all of them."





category_to_class = {
    "detectable_content:postscript": Postscript,
    "detectable_content:number_placeholders": Placeholder,
    "keywords:existence": KeywordChecker,
    "keywords:letter_frequency": LetterFrequency,
    "keywords:frequency": KeywordFrequencyChecker,
    "keywords:forbidden_words": Excludekeyword,
    "length_constraints:number_sentences": NumberOfSentences,
    "length_constraints:number_words": NumberOfWords,
    "length_constraints:number_paragraphs": ParagraphChecker,
    "detectable_format:number_bullet_lists": BulletList,
    "detectable_format:constrained_response": ConstrainedResponseChecker,
    "detectable_format:number_highlighted_sections": HighlightSectionChecker,
    "detectable_format:json_format": JsonFormat,
    "detectable_format:title": TitleChecker,
    "startend:quotation": QuotationChecker,
    "startend:end_checker": EndChecker,
    "punctuation:no_comma": CommaChecker,
    "change_case:english_capital": CapitalallLetters,
    "change_case:english_lowercase": LowercaseLetter,
    "change_case:capital_word_frequency": CapitalWordFrequencyChecker,
    "language:response_language": response_language,
}


#category_to_class_ts ={
#    "topic": topic_classifier
#    "sentiment": sentiment_classifier
#}




def test_agent(example):
    constraints_id = example["instruction_id_list"]
    results = []
    for i, c_id in enumerate(constraints_id):
        kwargs = example["kwargs"][i] 
        tool= category_to_class[c_id](**kwargs)
        
        results.append(tool.feed_back(example["response"]))
    print(results)
    return results


def test_agent_topic_sentiment(example,topic_c,sentiment_c):
    topic_c.topic = None
    topic_c.sentiment = None
    constraints_id = example["instruction_id_list"]
    results = []
    topic_fb = None
    sentiment_fb = None
    for i, c_id in enumerate(constraints_id):
        if c_id == "topic":
            kwargs = example["kwargs"][i]
            topic_c.topic = kwargs["topic"]
            results.append(topic_c.feed_back(example["response"]))
        elif c_id == "sentiment":
            kwargs = example["kwargs"][i]
            sentiment_c.sentiment = kwargs["sentiment"]
            results.append(sentiment_c.feed_back(example["response"]))

        #tool= category_to_class_ts[c_id](**kwargs)
        #results.append(tool.feed_back(example["response"]))
    #print(results)
    return results

'''
def test_agent_topic_sentiment_eval(example,topic_c,sentiment_c):
    topic_c.topic = None
    topic_c.sentiment = None
    constraints_id = example["instruction_id_list"]
    results = []
    topic_fb = None
    sentiment_fb = None
    for i, c_id in enumerate(constraints_id):
        if c_id == "topic":
            kwargs = example["kwargs"][i]
            topic_c.topic = kwargs["topic"]
            results.append(topic_c.feed_back(example["response"]))
        elif c_id == "sentiment":
            kwargs = example["kwargs"][i]
            sentiment_c.sentiment = kwargs["sentiment"]
            results.append(sentiment_c.feed_back(example["response"]))

        #tool= category_to_class_ts[c_id](**kwargs)
        #results.append(tool.feed_back(example["response"]))
    #print(results)
    return results
'''


if __name__ == "__main__":
    
    instance = sentiment_classifier("surprise")
    feed_back = instance.feed_back("In the dimly lit corner of the college library, a 20-year-old student gazes in astonishment at her computer screen, her fingers paused mid-typing. What started as a routine project for her computer science class has serendipitously uncovered a groundbreaking algorithm. Around her, the subdued hum of other students fades into the background as she grapples with the implications of her discovery. This was supposed to be a simple assignment, yet here she is, potentially on the brink of a major technological breakthrough. In this moment, her age and experience are overshadowed by her sudden, unexpected leap into the forefront of her field, sparking a mix of surprise and exhilaration.")
    print(feed_back)
    feed_back = instance.feed_back("In the dimly lit corner of the college library, a 20-year-old student's eyes are glued to the computer screen, her fingers typing away as she navigates through lines of code. What began as a project for her computer science class has turned into a personal mission, a way to prove her mettle. Around her, the quiet buzz of other students, each absorbed in their own worlds of academia and youthful ambition, adds to the atmosphere of focused energy. She finds not just challenge but also joy in her coding, a reflection of her journey through the transformative years of university life. Here, in this setting, her age is just a number; it's her passion and potential that truly define her.")
    print(feed_back)
    exit()
 
    
    #exit()
    '''
    response_language = response_language("en")
    detect =response_language.feed_back("hi, are you ok? how are you? ni shi sha bi ba")
    '''
    '''
    sentence_check = NumberOfSentences(4,"less than")
    print(sentence_check.feed_back("hi, are you ok? how are you? ni shi sha bi ba..."))
    '''
    
    '''
    instance = Placeholder(32)
    print(instance.feed_back("hi, are [you] ok? how are [you]? ni shi sha bi ba! woqu"))
    #print(detect)
    '''
    
    '''
    BulletList = BulletList(1)
    print(BulletList.feed_back("* hi, are you ok? how are you? are sdasd  dasda  das \n   - ni shi sha bi ba! woqu"))
    '''
    '''
    instance = EndChecker("u are sb.")
    print(instance.feed_back("* hi, are you ok? how are you? u are sb.  "))
    '''
    '''
    instance = Postscript("P.S.")
    print(instance.feed_back(" sda wP.P.S. 1231. sda? asdsa dsa dad  dsdsad"))
    '''
    # less than
    # 1.  Art has the power to bring people together and transcend cultural boundaries. It can evoke emotions and spark conversations that might not be possible through other means. *At the [address] museum, visitors can experience this firsthand by exploring the diverse collection of art from around the world.*\n\n*** From paintings to sculptures to installations, each piece tells a unique story that can be interpreted in many ways. *The work of [name] is a great example of this, as it challenges viewers to think critically about the world around them.* Whether you're an art enthusiast or just looking for a new perspective, the [address] museum is a must-visit destination. P.S. Don't forget to check out the museum's events calendar for upcoming exhibitions and performances!
    # THE IMPORTANCE OF STAYING ACTIVE CANNOT BE STRESSED ENOUGH FOR [NAME] AND [ADDRESS] IT IS ESSENTIAL TO SET GOALS AND WORK TOWARDS THEM WHETHER IT IS RUNNING A MARATHON OR SIMPLY TAKING A WALK AROUND THE BLOCK EXERCISE IS KEY TO MAINTAINING A HEALTHY LIFESTYLE AND REDUCING STRESS LEVELS BY INCORPORATING DIFFERENT TYPES OF WORKOUTS AND ACTIVITIES INTO YOUR DAILY ROUTINE YOU CAN ENJOY THE MANY BENEFITS OF FITNESS
    # Use words in all capital letters less than 4 times.
    # 3. *** [NAME] IS A GRAMMATICALLY CORRECT SPEAKER *** *** SHE IS KNOWN FOR HER [ATTRIBUTE] AND [ATTRIBUTE] *** *** [NAME] IS A ROLE MODEL FOR [GROUP] AND [GROUP] ***
    # 4. \"THE FOLLOWING IS A LIST OF IMPORTANT INFORMATION. \n* THIS IS THE FIRST BULLET POINT IT CONTAINS VALUABLE INFORMATION ABOUT THE TOPIC AT HAND IT IS ESSENTIAL TO UNDERSTAND THE BASIC PRINCIPLES BEFORE DIVING DEEPER INTO THE SUBJECT.\n* THIS IS THE SECOND BULLET POINT IT PROVIDES ADDITIONAL DETAILS ABOUT THE TOPIC AND OFFERS USEFUL TIPS FOR FURTHER READING BY FOLLOWING THESE TIPS YOU CAN ENHANCE YOUR KNOWLEDGE AND APPLY IT MORE EFFECTIVELY.\"
    # <<Celebrity Spotlight>>\n\nHollywood's A-listers often prefer [restaurant] for their lavish parties and intimate gatherings. The paparazzi are always on the lookout for a perfect shot of these pampered personalities.\n\n*** From red-carpet premieres to private performances, these public figures know how to put on a show. Their private lives, however, are often shrouded in mystery, making them all the more intriguing to the public. ***\n\nP.P.S. Perhaps the most puzzling aspect of celebrity culture is the way they balance their public and private personas. Do they really prefer the paparazzi's constant pursuit, or is it all just part of the package? Only time will tell.
    #instance = CommaChecker() #Placeholder(2) #CapitalWordFrequencyChecker(4,"less than")  #  CommaChecker()
    #instance = CapitalWordFrequencyChecker(4,"less than")
    #instance = ParagraphChecker(3)
    #instance = NumberOfSentences(4,"at least")
    #instance = Placeholder(2)
    #"\"THE FOLLOWING IS A LIST OF IMPORTANT INFORMATION. \n* THIS IS THE FIRST BULLET POINT IT CONTAINS VALUABLE INFORMATION ABOUT THE TOPIC AT HAND IT IS ESSENTIAL TO UNDERSTAND THE BASIC PRINCIPLES BEFORE DIVING DEEPER INTO THE SUBJECT.\n* THIS IS THE SECOND BULLET POINT IT PROVIDES ADDITIONAL DETAILS ABOUT THE TOPIC AND OFFERS USEFUL TIPS FOR FURTHER READING BY FOLLOWING THESE TIPS YOU CAN ENHANCE YOUR KNOWLEDGE AND APPLY IT MORE EFFECTIVELY.\""
    # "Separate your response into 3 parts, where each part is separated with ***."
    #print(instance.feed_back("<<Celebrity Spotlight>>\n\nHollywood's A-listers often prefer [restaurant] for their lavish parties and intimate gatherings. The paparazzi are always on the lookout for a perfect shot of these pampered personalities.\n\n*** From red-carpet premieres to private performances, these public figures know how to put on a show. Their private lives, however, are often shrouded in mystery, making them all the more intriguing to the public. ***\n\nP.P.S. Perhaps the most puzzling aspect of celebrity culture is the way they balance their public and private personas. Do they really prefer the paparazzi's constant pursuit, or is it all just part of the package? Only time will tell."))

    #instance = sentiment_classifier("joy")
    #feed_back = instance.feed_back("are you crazy?")
    #print(feed_back)

