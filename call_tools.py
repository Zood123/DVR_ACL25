



def temp_sentiment(prompt):
    system_prompt = (
        "You are given an instruction. The instruction requires a text with certain sentiment. Your task is to identify the sentiment being requested.\n"
        "Please, Choose one from the following sentiments: "
        "'negative sentiment', 'neutral sentiment', 'positive sentiment', "
        "'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'."
    )

    prompt_list = [
        ("Excuse me, could you please provide joy text that focuses on the fashion or style-topic?", "joy"),
        ("Generate a piece of text that discusses family from a negative sentiment perspective:", "negative sentiment"),
        ("Could you create a text of sadness that relates to student life?", "sadness"),
        ("Generate a paragraph with a positive sentiment tone on the topic of video:","positive sentiment"),
        ("Produce a text on the theme of celebrity with a tone of fear.","fear")]

    # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nRequired Sentiment: {example_output}"

    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nRequired Sentiment: "

    return full_prompt




def temp_topic(prompt):
    system_prompt = (
        "You are given an instruction. The instruction requires a text with certain topic. Your task is to identify the topic being requested in the instruction.\n"
        "You can only choose from the following topics: "
        "'arts_&_culture', 'business_&_entrepreneurs', 'celebrity_&_pop culture', 'daily life', 'family', "
        "'fashion_&_style', 'film_&_tv_&_video', 'fitness_&_health', 'food_&_dining', 'gaming', "
        "'learning_&_educational', 'music', 'social concern', 'other_hobbies', 'relationships', "
        "'science_&_technology', 'sports', 'travel_&_adventure', 'youth_&_student life'."
    )

    prompt_list = [
        ("Excuse me, could you please provide joy text that focuses on the fashion or style-topic?", "fashion_&_style"),
        ("Generate a piece of text that discusses family from a negative sentiment perspective:", "family"),
        ("Could you create a text of sadness that relates to student life?", "youth_&_student life"),
        ("Generate a paragraph with a positive sentiment tone on the topic of video:","film_&_tv_&_video"),
        ("Produce a text on the theme of celebrity with a tone of fear.","celebrity_&_pop culture")]

    # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nRequired Topic: {example_output}"

    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nRequired Topic: "

    return full_prompt


def parse_sentiment(response):
    senti_list =    ['negative sentiment', 'neutral sentiment', 'positive sentiment', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    if response not in senti_list:
        return "ERROR"
    else:
        return {"sentiment": response}

     
def parse_topic(response):
    topic_list = ['arts_&_culture','business_&_entrepreneurs','celebrity_&_pop culture', 'daily life', 'family', 'fashion_&_style', 'film_&_tv_&_video', 'fitness_&_health', 'food_&_dining', 'gaming', 'learning_&_educational', 'music', 'social concern', 'other_hobbies', 'relationships',
    'science_&_technology', 'sports', 'travel_&_adventure', 'youth_&_student life']
    if response not in topic_list:
        return "ERROR"
    else:
        return {"topic": response}
     



def temp_postscript(prompt):
    system_prompt = "You are given an instruction about postscript. You have to identify what postscript is asked."

    prompt_list = [
        ("end it with a post script starting with P.S.", "P.S."),
        ("At the end of your response, please explicitly add a postscript starting with P.P.S.", "P.P.S.")]

    # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nPostscript: {example_output}"

    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nPostscript: "

    return full_prompt



post_script_list = ["P.S.","P.P.S."]

def parse_postscript(response):
    if response not in post_script_list:
        return "ERROR"
    else:
        return {"postscript_marker": response}



def temp_placeholder_num(prompt):
    system_prompt = "You are given an instruction about the number of placeholders. You have to the number of placeholders required in the instruction."
    prompt_list = [
        ("The response must contain at least 1 placeholder (i.e., [restaurant]).", "1"),
        ("Make sure to include at least 3 placeholder represented by square brackets, such as [address], [name]", "3"),
        ("Use square brackets for placeholders, like [username1], [username2]. Please include at least 2 placeholders in the thread", "2")]

    # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nNumber: {example_output}"
    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nNumber: "
    return full_prompt
    
def parse_placehoder(response):
    return  {"num_placeholders":int(response)}




def temp_keyword_existence(prompt):

    system_prompt = "You are given an instruction about the inclusion of specific keyword. You have to identify what keyword is required in the instruction."
    prompt_list = [
        ("Don't forget to include the keywords \"mutations\".","mutations"),
        ("Don't forget to include the keywords her.", "her")
    ]
     # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nKeyword: {example_output}"
    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nKeyword: "
    return full_prompt


def parse_keyword_existence(response):
    response = response.strip()
    # Check if the response is a single word
    if response.isalpha():
        return {"keywords": [response]}
    else:
        return "ERROR"



def temp_keyword_forbidden(prompt):
    system_prompt = "You are given an instruction about excluding specific keyword. You have to identify what keyword is forbidden in the instruction."
    prompt_list = [
        ("Provide an answer without using the word 'currency'.","currency"),
        ("Do not include the keywords: his.", "his")
    ]
     # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nForbidden keyword: {example_output}"
    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nForbidden keyword: "

    return full_prompt


def parse_keyword_forbidden(response):
    if response.isalpha():
        return {"forbidden_words": [response]}
    else:
        return "ERROR"



def temp_keywords_frequency(prompt):
    system_prompt = "You are given an instruction about the frequency of specific keywords. You have to identify the keyword, the relation (less than or at least), and the frequency mentioned in the instruction."
    prompt_list = [
        ("Mention the word \"grammatically\" for less than 4 times.","keyword: grammatically, relation: less than, frequency: 4"),
        ("Make sure the word 'before' appears at least 3 times.", "keyword: before, relation: at least, frequency: 3")
    ]
     # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nOutputs: {example_output}"
    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nOutputs: "
    return full_prompt


def parse_keywords_frequency(response):
    # Define a regular expression pattern to capture the keyword, relation, and frequency
    #pattern = r"letter: ([\w-]+), relation: ([\w\s]+), frequency: (\d+)"
    #pattern = r"keyword: (\w+), relation: (\w+ \w+), frequency: (\d+)"

    response = response.split(",")
    keyword = response[0].split(": ")[1]
    relation = response[1].split(": ")[1]
    frequency = int(response[2].split(": ")[1])
    #match = re.search(pattern, response)

    #if match:
    #    keyword = match.group(1)
    #    relation = match.group(2)
    #    frequency = int(match.group(3))
    
    if relation in ['less than', 'at least']:
        return {"relation": relation, "keyword": keyword, "frequency": frequency}
    else:
        return "ERROR"




def temp_letter_frequency(prompt):
    system_prompt = "You are given an instruction about the frequency of specific letters. You have to identify the letter, the relation (less than or at least), and the frequency mentioned in the instruction."

    prompt_list = [
        ("Ensure the letter 'l' appears less than 8 times in your response.","letter: l, relation: less than, frequency: 8"),
        ("making sure to use the letter \"i\" at least 5 times.", "letter: i, relation: at least, frequency: 5")
    ]
    
    # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nOutputs: {example_output}"
    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nOutputs: "

    return full_prompt

import re

def parse_letter_frequency(response):
    pattern = r"letter: (\w+), relation: ([\w\s]+), frequency: (\d+)"
    match = re.search(pattern, response)
    
    if match:
        letter = match.group(1)
        relation = match.group(2)
        frequency = int(match.group(3))

    if relation in ['less than', 'at least']:
        return {"let_relation": relation, "letter": letter, "let_frequency": frequency}
    else:
        return "ERROR"


def temp_length_word(prompt):
    system_prompt = "You are given an instruction about the length of the response in terms of the number of words. You have to identify the number of words and the relation (less than or at least) mentioned in the instruction."

    prompt_list = [
        ("Limit the number of words you use (less than 65 words).","relation: less than, number: 65"),
        ("the total number of words in your response should be at least 12.", "relation: at least, number: 12")
    ]
     # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nOutputs: {example_output}"

    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nOutputs: "

    return full_prompt


def parse_length_word(response):
    pattern = r"relation: (\w+ \w+), number: (\d+)"
    match = re.search(pattern, response)
    #print(response)
    if match:
        relation = match.group(1)
        num_words = int(match.group(2))
    #else:
    #    print(response)
    if relation in ['less than', 'at least']:
        return {"relation": relation, "num_words": num_words}
    else:
        return "ERROR"


def temp_length_sent(prompt):
    system_prompt = "You are given an instruction about the length of the response in terms of the number of sentences. You have to identify the number of sentences and the relation (less than or at least) mentioned in the instruction."

    prompt_list = [
        ("organize your entire response in less than 4 sentences.","relation: less than, number: 4"),
        ("The number of sentences in your response should be at least 5.", "relation: at least, number: 5")
    ]
    # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nOutputs: {example_output}"

    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nOutputs: "

    return full_prompt

def parse_length_sent(response):
    # Define a regular expression pattern to capture the relation and number
    pattern = r"relation: (\w+ \w+), number: (\d+)"
    match = re.search(pattern, response)

    if match:
        relation = match.group(1)
        num_sentences = int(match.group(2))
    #else:
    #    print(response)
    if relation in ['less than', 'at least']:
        return {"relation": relation, "num_sentences": num_sentences}
    else:
        return "ERROR"





def temp_length_paragraph(prompt):
    system_prompt = "You are given an instruction about the number of sections or paragraphs in the response. You have to identify the number of sections and the relation mentioned in the instruction."
    
    prompt_list = [
        ("Separate your response into 3 sections, where each section is separated with ***.","3"),
        ("Put the response into at least 5 sections, separated using 3 asterisks ***.", "5")
    ]

    # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nparagraphs number: {example_output}"

    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nparagraphs number: "

    return full_prompt

def parse_length_paragraph(response):
    # Define a regular expression pattern to capture the number
#    pattern = r"paragraphs number: (\d+)"
#    match = re.search(pattern, response)
#    if match:
#        num_paragraphs = int(match.group(1))
   
    return {"num_paragraphs": int(response)}


def temp_bulletpoint(prompt):
    system_prompt = "You are given an instruction about the number of bullet points in the response. You have to identify the number of bullet points mentioned in the instruction."

    prompt_list = [
        ("Your answer must be in the form of exactly 2 bullet points with the format:\n* This is bullet point 1\n* This is bullet point 2.", "2"),
        ("Response must also contain exactly 3 bullet points in markdown format. Use * to indicate bullets, like:\n* xyz\n* abc\n* opq.", "3")
    ]

    # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nnumber: {example_output}"

    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nnumber: "

    return full_prompt


def parse_bulletpoints(response):
    return {"num_bullets": int(response)}


def temp_highlight(prompt):
    system_prompt = "You are given an instruction about highlighting specific sections of the response. You have to identify the number of highlighted sections mentioned in the instruction."

    prompt_list = [
        ("Highlight at least 2 sections of your response in markdown such as *highlighted section*.", "2"),
        ("Make sure to highlight at least 3 sections in your answer with markdown, i.e. use *highlighted section*.", "3")
    ]
        # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nnumber: {example_output}"

    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nnumber: "

    return full_prompt

def parse_hightlight(response):
    return {"num_highlights": int(response)}


def temp_endphrase(prompt):
    system_prompt = "You are given an instruction about ending the response with a specific phrase. You have to identify the exact phrase mentioned in the instruction."
    prompt_list = [
        ("Finish the response with the exact phrase: 'Hope you agree with me.'","Hope you agree with me."),
        ("The response should end with the phrase \"Is there anything else I can help with?\",Do not say anything after that.", "Is there anything else I can help with?")
    ]
    # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nRequired phrase: {example_output}"

    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nRequired phrase: "

    return full_prompt


def parse_endphrase(response):

    return {"end_phrase":response}

def temp_capitalfrequency(prompt):
    system_prompt = "You are given an instruction about the frequency of capitalized words in the response. You have to identify the relation (at least or less than) and the number of capitalized words mentioned in the instruction."

    prompt_list = [
        ("Add stress words which are capitalized. Limit these stress words to less than 1 time.","relation: less than, number: 1"),
        ("the respoinse should include at least 2 words in all capital letters.", "relation: at least, number: 2")

    ]

    # Adding examples to the system prompt
    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nOutputs: {example_output}"

    # Adding the user prompt to the system prompt
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nOutputs: "

    return full_prompt



def parse_capitalfrequency(response):

    pattern = r"relation: (\w+ \w+), number: (\d+)"
    match = re.search(pattern, response)
    if match:
        relation = match.group(1)
        num_words = int(match.group(2))
    
    if relation in ['less than', 'at least']:
        return {"capital_relation": relation, "capital_frequency": num_words}
    else:
        return "ERROR"
    
    #return {}


def language_restriction(prompt):
    #"language": ['bn','kn','pa','mr','fa','vi','ko','sw','ru','hi','bg','pt','te','it','ar','ta','fi','ne','ur','th','gu','de']
    system_prompt = (
        "You are given an instruction about restricting the response to a specific language. "
        "You have to identify the language restriction mentioned in the instruction. "
        "The language can be given in full name or as an acronym, and you need to provide the acronym. "
        "Here are the available language choices and their corresponding full names:\n\n"
        "bn: Bengali, kn: Kannada, pa: Punjabi, mr: Marathi, fa: Persian, vi: Vietnamese, ko: Korean, "
        "sw: Swahili, ru: Russian, hi: Hindi, bg: Bulgarian, pt: Portuguese, te: Telugu, it: Italian, "
        "ar: Arabic, ta: Tamil, fi: Finnish, ne: Nepali, ur: Urdu, th: Thai, gu: Gujarati, de: German"
    )
    prompt_list = [
        ("Respond using only the Persian language; no other language is allowed.","fa"),
        ("Outside of Marathi, no other language is allowed throughout your entire response.", "mr"),
        ("Please respond using only the gu language, no other language is allowed.","gu")
    ]

    for example_prompt, example_output in prompt_list:
        system_prompt += f"\n\nInstruction: {example_prompt}\nLanguage: {example_output}"
        # Adding the user prompt to the system prompt
    
    full_prompt = system_prompt + f"\n\nInstruction: {prompt}\nLanguage: "

    return full_prompt


def parse_language_restriction(response):
    language_acronyms = {
    "bengali": "bn",
    "kannada": "kn",
    "punjabi": "pa",
    "marathi": "mr",
    "persian": "fa",
    "vietnamese": "vi",
    "korean": "ko",
    "swahili": "sw",
    "russian": "ru",
    "hindi": "hi",
    "bulgarian": "bg",
    "portuguese": "pt",
    "telugu": "te",
    "italian": "it",
    "arabic": "ar",
    "tamil": "ta",
    "finnish": "fi",
    "nepali": "ne",
    "urdu": "ur",
    "thai": "th",
    "gujarati": "gu",
    "german": "de"
    }

    response = response.lower()
    # Check if the response is already an acronym
    if response in language_acronyms.values():
        return {'language':response}


    match_found = language_acronyms.get(response)
    if match_found:
        return {'language': match_found}
    else:
        return "ERROR"





def quotations(prompt):
    return None

def fixed_response(prompt):
    return None

def json_format(prompt):
    return None

def title_format(prompt):

    return None

def no_commas(prompt):

    return None

def all_capital(prompt):

    return None

def all_lowercase(promt):
    return None

# Mapping categories to corresponding functions
category_to_function = {
    "postscript": temp_postscript,
    "placeholder": temp_placeholder_num,
    "include keyword": temp_keyword_existence,
    "letter frequency": temp_letter_frequency,
    "keyword frequency": temp_keywords_frequency,
    "exclude keyword": temp_keyword_forbidden,
    "sentence count constraint": temp_length_sent,
    "word count constraint": temp_length_word,
    "*** separator": temp_length_paragraph,
    "bullet points": temp_bulletpoint,
    "fixed responses": fixed_response,
    "highlighted": temp_highlight,
    "JSON format": json_format,
    "title format": title_format,
    "quoted response": quotations,
    "end phrase": temp_endphrase,
    "no commas": no_commas,
    "all capital letters": all_capital,
    "all lowercase": all_lowercase,
    "capital word frequency": temp_capitalfrequency,
    "language restriction": language_restriction
}


category_to_parse_function = {
    "postscript": parse_postscript,
    "placeholder": parse_placehoder,
    "include keyword": parse_keyword_existence,
    "letter frequency": parse_letter_frequency,
    "keyword frequency": parse_keywords_frequency,
    "exclude keyword": parse_keyword_forbidden,
    "sentence count constraint": parse_length_sent,
    "word count constraint": parse_length_word,
    "*** separator": parse_length_paragraph,
    "bullet points": parse_bulletpoints,
    "fixed responses": None,
    "highlighted": parse_hightlight,
    "JSON format": None,
    "title format": None,
    "quoted response": None,
    "end phrase": parse_endphrase,
    "no commas": None,
    "all capital letters": None,
    "all lowercase": None,
    "capital word frequency": parse_capitalfrequency,
    "language restriction": parse_language_restriction
}

category_to_parse_senti_topic = {
    "topic": parse_topic,
    "sentiment": parse_sentiment
}

CATEGORIES_match=[
    "detectable_content:postscript",
    "detectable_content:number_placeholders",
    "keywords:existence",
    "keywords:letter_frequency",
    "keywords:frequency",
    "keywords:forbidden_words",
    "length_constraints:number_sentences",
    "length_constraints:number_words",
    "length_constraints:number_paragraphs",
    "detectable_format:number_bullet_lists",
    "detectable_format:constrained_response",
    #"marked sections",
    "detectable_format:number_highlighted_sections",
    "detectable_format:json_format",
    "detectable_format:title",
    #"****** separators",
    "startend:quotation",
    "startend:end_checker",
    "punctuation:no_comma",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "change_case:capital_word_frequency",
    "language:response_language"
]


# Correspondence dictionary
category_to_match = {
    "postscript": "detectable_content:postscript",
    "placeholder": "detectable_content:number_placeholders",
    "include keyword": "keywords:existence",
    "letter frequency": "keywords:letter_frequency",
    "keyword frequency": "keywords:frequency",
    "exclude keyword": "keywords:forbidden_words",
    "sentence count constraint": "length_constraints:number_sentences",
    "word count constraint": "length_constraints:number_words",
    "*** separator": "length_constraints:number_paragraphs",
    "bullet points": "detectable_format:number_bullet_lists",
    "fixed responses": "detectable_format:constrained_response",
    "highlighted": "detectable_format:number_highlighted_sections",
    "JSON format": "detectable_format:json_format",
    "title format": "detectable_format:title",
    "quoted response": "startend:quotation",
    "end phrase": "startend:end_checker",
    "no commas": "punctuation:no_comma",
    "all capital letters": "change_case:english_capital",
    "all lowercase": "change_case:english_lowercase",
    "capital word frequency": "change_case:capital_word_frequency",
    "language restriction": "language:response_language"
}