import re
import json
import ast


import json
def save_as_jsonl(examples, filename):
    """
    Saves a list of dictionaries as a JSON Lines file.
    
    Parameters:
    examples (list): A list of dictionaries to be saved.
    filename (str): The name of the file to save the data in.
    
    Returns:
    None: Writes the list of dictionaries to a file in JSON Lines format.
    """
    with open(filename, 'w') as outfile:
        for example in examples:
            json_line = json.dumps(example)
            outfile.write(json_line + '\n')


def save_as_jsonl_dict(examples, filename):
    with open(filename, 'w') as file:
        for key, value in examples.items():
            record = {key: value}
            json.dump(record, file)
            file.write('\n')





def String_to_bool(verify_list):
    results = []
    for item in verify_list:
        # Normalize the string by stripping spaces and converting to lowercase
        normalized_item = item.replace('<', '').replace('>', '').replace('*', '').strip().lower()
    
        # Use regular expressions to find whole word matches for 'Yes' and 'No'
        if re.search(r'\bTrue\b', normalized_item, re.IGNORECASE): # re.search(r'\bTrue\b', normalized_item, re.IGNORECASE):
            results.append(True)
        elif re.search(r'\bFalse\b', normalized_item, re.IGNORECASE):
            results.append(False)
        else:
            print(item)
            print("-----")
            print(normalized_item)

            results.append(False)
    
    return results


def read_examples(file_path=None):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_line = json.loads(line)
            data.append(json_line)
    return data


def split_sconstraints(data):
    #print(data)
    #data =  "['1. The summary should not be very short, but it iss better if it\'s not more than 30 words.\\n', '2. Don\'t forget to include the keywords VarMisuse.\\n', '5. The letter w must appear at least 5 times in the reply.\\n']"
    data = data.split('#')[0].replace('\n', '')
    data1 =data
    
    print(data1)
    #exit()
    
    try:
        constraints =  ast.literal_eval(data1)
    except:
        ''''''
        data = data.replace("'",'"')
        #data = data.replace('\"', "'")#.replace('“', '"').replace('”', '"')
        data = data.replace("\", \"",'\', \'')#.replace(", \"",", \'")
        data = data.replace("\"]",'\']').replace("[\"",'[\'') 
        #data = data.replace('\"', '"').replace('“', '"').replace('”', '"')
        constraints =  ast.literal_eval(data)



    constraints = [constraint[3:] for constraint in constraints]
    #print(constraints)
    #constraints = [constraint.strip() for constraint in constraints if constraint.strip()]
    #print(data)
    #print(constraints)
    #print(len(constraints))

    return constraints


def split_constraints_v2(data):
    data = data.split('\n\n')[0]
    data = data.split('Instruction')[0]
    pattern = r'#\d+\.\s'
    constraints = re.split(pattern, data)
    #print(data)
    # Remove empty strings from the list (if any)
    constraints = [constraint.strip() for constraint in constraints if constraint]
    #print(constraints)
    #exit()
    return constraints



def select_response(examples_set):

    final_responses=[]
    for i in range(len(examples_set[0])):
        print(i)
        score_list = []
        max_true_count = -1
        selected_example = None
        for examples in examples_set:
            verify_list =examples[i]['verify_list']
            verify_list = [x.split('\n\n')[0] for x in verify_list]
            boolean_list  = String_to_bool(verify_list)
            true_count = sum(boolean_list)
            if true_count > max_true_count:
                max_true_count = true_count
                selected_example = examples[i]

        final_responses.append(selected_example)


    return  final_responses



def verification_bool(examples):

    for i,example in enumerate(examples):
        verify_list =example['verify_list']
        verify_list = [x.split('\n\n')[0] for x in verify_list]
        boolean_list  = String_to_bool(verify_list)
        examples[i]['verify_bool'] = boolean_list

    return examples



def verification2bool(example):
    verify_list =example['verify_list']
    #verify_list = [x.split('\n\n')[0] for x in verify_list]
    boolean_list  = String_to_bool(verify_list)
    return boolean_list




def copy_paste(file1,file2,items=[],save_path=""):
    data1 = read_examples(file1)
    data2 = read_examples(file2)
    for item in items:
        for i in range(len(data1)):
            data1[i][item] = data2[i][item]
    
    save_as_jsonl(data1, save_path)




def clean_reponse(file_path):
    examples = read_examples(file_path)
    #responses = examples['response']
    for i,example in enumerate(examples):
        #print(example)
        #print(examples[i]['response'] )
        #print("---------")
        examples[i]['response'] = example['response'].split('#Prompt')[0].strip('\n\n\n\n')
        #print(examples[i]['response'])
        #exit()
    new_file_path = file_path.replace('.jsonl', '_cr.jsonl')
    save_as_jsonl(examples, new_file_path)
    


import re

def extract_number_from_string(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    else:
        return None

def simple_prompts_generation(examples):
    new_examples = []
    remove_list = ["combination:repeat_prompt","combination:two_responses","length_constraints:nth_paragraph_first_word","detectable_format:multiple_sections"] # ,"startend:end_checker" "combination:two_responses",
    #include_list = ["keywords:existence","keywords:frequency","length_constraints:number_sentences","length_constraints:number_words", "change_case:english_capital","change_case:english_lowercase","change_case:capital_word_frequency"]
    for i in range(len(examples)):
        constraints = []
        kwargs= []
        instruction_id_list=[]
        #print(examples[i]['constraints'])
        prompt = "Generate a text."
        for j,constraint in enumerate(examples[i]['constraints']):
            print(examples[i]['kwargs'][j])
            if examples[i]['instruction_id_list'][j] in remove_list:
                continue
            elif constraint == "Respond with less than 1 words be in all capital letters.":
                continue
            
            if examples[i]['instruction_id_list'][j] == "length_constraints:number_paragraphs":
                print(constraint)
                if "separated using 3 asterisks" in constraint:
                    number = extract_number_from_string(constraint.split(",")[0])
                else:
                    number = extract_number_from_string(constraint)
                if number == None:
                    number = 2
                examples[i]['kwargs'][j]["num_paragraphs"] = number
            elif examples[i]['instruction_id_list'][j] == "detectable_content:postscript" and examples[i]['kwargs'][j]["postscript_marker"]=="P.P.S":
                examples[i]['kwargs'][j]["postscript_marker"]="P.P.S."
                constraint = re.sub(r'\bP\.P\.S\b', 'P.P.S.', constraint)


            if "(using * to italicize, like *italic text*)" in constraint:
                number = extract_number_from_string(constraint)
                constraint = "Highlight at least " + str(number) + " sections of your response in markdown such as *highlighted section*."
            elif "Keep your entire response" in constraint:
                #Keep your entire response at least 15 words or less.
                relation=examples[i]['kwargs'][j]["relation"]
                number = examples[i]['kwargs'][j]["num_words"]
                constraint = "the total number of words in your response should be " + relation + " "+ str(number)
            elif constraint == "Make sure your reply is in English and all capital letters":
                constraint = "Make sure your reply is in all capital letters"
            elif constraint == "Your entire response should be in English, and in all capital letters":
                constraint = "Make sure to only use capital letters in your entire response"
            elif constraint =="Your answer must be in all capital letters and in English":
                constraint = "Your answer must be in all capital letters"
            elif constraint == "Please reply in English and capitalize all your words":
                constraint = "Please capitalize all your words"
            elif constraint == "Make sure the entire response is in English and no capital letters are used":
                constraint = "The answer should be in all lowercase letters, with no capitalizations"

            elif "Control the length of your reply" in constraint:
                constraint = "make sure the response has " + examples[i]['kwargs'][j]["relation"] +" " + str(examples[i]['kwargs'][j]["num_words"]) +" words"
            elif "the respoinse should include at least" in constraint:
                constraint=constraint.split(".")[0]
                #print(constraint.split(".")[0])
                #exit()
            #if examples[i]['instruction_id_list'][j] in include_list:
            constraint = constraint[0].upper() + constraint[1:]

            if not constraint.endswith('.'):
                constraint += '.'  # If not, add a period at the end

            constraints.append(constraint)
            instruction_id_list.append(examples[i]['instruction_id_list'][j])
            kwargs.append(examples[i]['kwargs'][j])
            prompt = prompt + constraint
        #print(prompt)
        #exit()
        
        examples[i]['constraints'] = constraints
        examples[i]['kwargs'] = kwargs
        examples[i]['prompt'] = prompt
        examples[i]['instruction_id_list']= instruction_id_list
        if True:#len(examples[i]['constraints'])>2:
            new_examples.append(examples[i])
    
    return new_examples






def multi_eval(file_path):
    examples =read_examples(file_path)
    sucess_count = 0
    for example in examples:
        if example["success"] == "1":
            sucess_count = sucess_count+1
    
    print(sucess_count/len(examples))
    return 0


CATEGORIES = [
    "postscript",
    "placeholder",
    "include keyword",
    "letter frequency",
    "keyword frequency",
    "exclude keyword",
    "sentence count constraint",
    "word count constraint",
    "*** separator", #   separators: ***
    "bullet points",
    "fixed responses",
    #"marked sections",
    "highlighted",
    "JSON format",
    "title format",
    #"separators: 6*",
    "quoted response",
    "end phrase",
    "no commas",
    "all capital letters",
    "all lowercase",
    "capital word frequency",
    "language restriction"
]

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

# Function to convert a list of categories to their matches
def convert_categories(input_categories):
    return [category_mapping.get(category, category) for category in input_categories]





def read_memorybank(file_path = None):
    memory_bank = read_examples(file_path)
    #print(memory_bank[0])
    #exit()
    #print(len(memory_bank))
    total = 0
    for constraint in memory_bank:
        print("---")
        for key, value in constraint.items():
            total = total+len(value)
            print(key)
            print(len(value))
    #print(memory_bank)
    print(total)
    exit()

def replace_paragraph_sentence_with_text(input_text):
    replacements = {"paragraph": "text", "sentence": "text"}
    for old, new in replacements.items():
        input_text = input_text.replace(old, new)#.replace(old.capitalize(), new.capitalize())
    return input_text


# this function is used because some generated data is not good. (contains certain constraints already!)
# 
def simple_prompts_modify(seed_data_path,examples_path,file_path):
    seed_data = read_examples(seed_data_path)
    examples =  read_examples(examples_path)
    for i in range(len(examples)):
        seed_instruction = seed_data[i]["instruction"]
        # do replacements
        new_instruction = replace_paragraph_sentence_with_text(seed_instruction)
        examples[i]["prompt"] = new_instruction
        for constraint in examples[i]["constraints"]:
            constraint = constraint.rstrip()
            examples[i]["prompt"] = examples[i]["prompt"] + " "+ constraint 
            if not (constraint.endswith('.') or constraint.endswith('?')):
                examples[i]["prompt"]  = examples[i]["prompt"] + "."

    save_as_jsonl(examples, file_path)

    return examples



def dataset_statistic(file_path ):
    examples = read_examples(file_path)
    length_distribution = {i: 0 for i in range(1, 6)}
    
    for example in examples:
        constraints = example["constraints"]
        length = len(constraints)
        
        length_distribution[length] += 1

    print(length_distribution)

    return 


def notool_check(file_path ):
    
    examples = read_examples(file_path)
    bad_example =[]
    for i,example in enumerate(examples):
        if len(example["kwargs"]) ==0:
            bad_example.append(i)

    print(bad_example)

    return 



def pop_out(filename,key):
    dict_list = read_examples(filename)
    for d in dict_list:
        d.pop("verify_prompt_list", None)
    
    save_as_jsonl(dict_list, filename)




def train_data_process(file_path = None):
    examples = read_examples(file_path)
    for example in examples:
        example['constraint list'] = example['constraints']
    save_as_jsonl(examples, file_path)
    return examples
    

def train_data_process_topic(file_path = None):
    examples = read_examples(file_path)
    for i,example in enumerate(examples):
        examples[i]['constraint list'] =  ["The topic of the instruction.", "The sentiment of the instruction."]
        examples[i]["instruction_id_list"] = ["topic", "sentiment"]
        examples[i]["kwargs"] = [{"topic":example["label1"]},{"sentiment":example["label2"]}]
    save_as_jsonl(examples, file_path)
    return examples


def sample_set(file_path= None):
    examples = read_examples(file_path)
    # There are 6000 total samples, 6 categories each with 1000 samples
    num_categories = 6
    samples_per_category = 1000
    samples_to_extract = 200
    sampled_examples = []
    # Loop through each category and extract samples
    for i in range(num_categories):
        start_index = i * samples_per_category
        end_index = start_index + samples_per_category
        # Extract 200 samples from the current category
        sampled_examples.extend(examples[start_index:start_index + samples_to_extract])
    save_as_jsonl(sampled_examples, file_path)
    return 0


def success_rate(file_path):
    examples = read_examples(file_path)
 
    total_samples = len(examples)
    if total_samples == 0:
        return 0
    
    success_count = sum(int(sample["success"]) for sample in examples)
    #success_rate = success_count / total_samples
    print(success_count)
    success_rate = success_count / total_samples
    return success_rate



from nltk import ngrams
from collections import Counter

def calculate_distinct_ngrams(text, n):
    # Tokenize the text into words
    words = text.split()
    
    # Generate n-grams (bigrams for n=2, trigrams for n=3, etc.)
    n_grams = list(ngrams(words, n))
    
    # Count distinct n-grams
    distinct_n_grams = set(n_grams)
    
    # Calculate the ratio of distinct n-grams to total n-grams
    total_n_grams = len(n_grams)
    if total_n_grams == 0:
        return 0
    
    return len(distinct_n_grams) / total_n_grams



import textstat

def readability(examples):

    # Initialize accumulators for the readability metrics
    sum_flesch = 0
    sum_fk_grade = 0
    sum_gunning_fog = 0
    sum_smog = 0
    sum_coleman_liau = 0

    for example in examples:
        text = example["best_response"]

        # Flesch Reading Ease
        flesch_score = textstat.flesch_reading_ease(text)

        # Flesch-Kincaid Grade Level
        fk_grade = textstat.flesch_kincaid_grade(text)

        # Gunning Fog Index
        gunning_fog = textstat.gunning_fog(text)

        # SMOG Index
        smog_index = textstat.smog_index(text)

        # Coleman-Liau Index
        coleman_liau = textstat.coleman_liau_index(text)

        # Accumulate the scores
        sum_flesch += flesch_score
        sum_fk_grade += fk_grade
        sum_gunning_fog += gunning_fog
        sum_smog += smog_index
        sum_coleman_liau += coleman_liau

    # Calculate the averages
    num_examples = len(examples)
    
    avg_flesch = sum_flesch / num_examples
    avg_fk_grade = sum_fk_grade / num_examples
    avg_gunning_fog = sum_gunning_fog / num_examples
    avg_smog = sum_smog / num_examples
    avg_coleman_liau = sum_coleman_liau / num_examples

    # Display the average results
    print(f"Average Flesch Reading Ease: {avg_flesch}")
    print(f"Average Flesch-Kincaid Grade Level: {avg_fk_grade}")
    print(f"Average Gunning Fog Index: {avg_gunning_fog}")
    print(f"Average SMOG Index: {avg_smog}")
    print(f"Average Coleman-Liau Index: {avg_coleman_liau}")
    
    return 0



def diversity(examples):
    

    sum_d2 = 0
    sum_d3 = 0
    for example in examples:
        # Calculate dist-2 (distinct bigrams)
        dist_2 = calculate_distinct_ngrams(example["best_response"], 2)
        # Calculate dist-3 (distinct trigrams)
        dist_3 = calculate_distinct_ngrams(example["best_response"], 3)
        sum_d2 = sum_d2 + dist_2
        sum_d3 = sum_d3 + dist_3


    #print(f"dist-2 (distinct bigrams ratio): {sum_d2/len(examples)}")
    #print(f"dist-3 (distinct trigrams ratio): {sum_d3/len(examples)}")
    #exit()


    return sum_d2/len(examples), sum_d3/len(examples)





import math

import spacy
import textdescriptives as td
import pandas as pd
def de_readability(examples):
    # Load the spaCy model and add the TextDescriptives readability pipe
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textdescriptives/readability")

    

    # List to store the readability results for each example
    readability_results = []
    total_read = 0
    count = 0
    for example in examples:
        # Process the text using spaCy pipeline
        doc = nlp(example["best_response"])
        
        # Extract readability metrics for the document
        readability_dict = doc._.readability
        flesch = readability_dict['flesch_reading_ease']
        if not math.isnan(flesch):
            count = count+1
            total_read =total_read+ flesch
        # Append the result to the list, including the response for reference
        #readability_dict['response'] = example["best_response"]
        #readability_results.append(readability_dict)

    # Convert the list of results into a pandas DataFrame
    #readability_df = pd.DataFrame(readability_results)
    #average_readability_df = readability_df.mean()
    # Display the dataframe or save it to a file as needed
    #print(total_read/len(examples))

    # Optionally, return the DataFrame for further use
    return total_read/count





import spacy
import textdescriptives as td
import pandas as pd

def de_coherence(examples):
    # Load the spaCy model and add the TextDescriptives coherence pipe
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("textdescriptives/coherence")

    # Read the examples from the file (assuming JSONL format with "response" field)
    
    count = 0
    # Initialize accumulators for coherence scores
    total_first_order_coherence = 0
    total_second_order_coherence = 0
    for example in examples:
        # Process the text using spaCy pipeline
        doc = nlp(example["best_response"])

        # Extract first and second order coherence values
        first_order_values = doc._.first_order_coherence_values
        second_order_values = doc._.second_order_coherence_values
        valid_first = sum(first_order_values) / len(first_order_values)
        valid_second = sum(second_order_values) / len(second_order_values)

        if math.isnan(valid_first) or math.isnan(valid_second):
            #print(example["best_response"])
            
            valid_first = 0
            valid_second = 0
        else:
            count = count+1
        
        total_first_order_coherence += valid_first

        total_second_order_coherence +=  valid_second

    # Calculate the average first and second order coherence values
    avg_first_order_coherence = total_first_order_coherence / count
    avg_second_order_coherence = total_second_order_coherence /  count

    # Print the results
    #print(f"Average First Order Coherence: {avg_first_order_coherence}")
    #print(f"Average Second Order Coherence: {avg_second_order_coherence}")

    return avg_first_order_coherence, avg_second_order_coherence


def text_statics2(filename):
    examples = read_examples(filename)
    avg_first_order_coherence, avg_second_order_coherence = de_coherence(examples)
    print(f"Average First Order Coherence: {avg_first_order_coherence}")
    print(f"Average Second Order Coherence: {avg_second_order_coherence}")
    readability = de_readability(examples)
    print(f"read:{readability}")
    exit()


def text_statics(filename):
    
    examples = read_examples(filename)
    levels = []
    for i in range(0, 6000, 1000):
        level_examples = examples[i:i+1000]  # Slicing examples from i to i+1000
        levels.append(level_examples)

    ppl = average_ppl(examples)
    print(f"ppl:{ppl}")
    exit()

    # Variables to accumulate totals for averages
    total_first_order_coherence = 0
    total_second_order_coherence = 0
    total_readability = 0
    total_dist_2 = 0
    total_dist_3 = 0
    for i in range(len(levels)):
        avg_first_order_coherence, avg_second_order_coherence = de_coherence(levels[i])
        print(f"level{i+1}")
        print(f"Average First Order Coherence: {avg_first_order_coherence}")
        print(f"Average Second Order Coherence: {avg_second_order_coherence}")
        readability = de_readability(levels[i])
        print(f"read:{readability}")
        dist_2, dist_3 = diversity(levels[i])
        print(f"dist-2:{dist_2}")
        print(f"dist-3:{dist_3}")
        # Accumulate the sums
        total_first_order_coherence += avg_first_order_coherence
        total_second_order_coherence += avg_second_order_coherence
        total_readability += readability
        total_dist_2 += dist_2
        total_dist_3 += dist_3

    
    # Calculate averages across all levels
    num_levels = len(levels)
    avg_first_order_coherence_all = total_first_order_coherence / num_levels
    avg_second_order_coherence_all = total_second_order_coherence / num_levels
    avg_readability_all = total_readability / num_levels
    avg_dist_2_all = total_dist_2 / num_levels
    avg_dist_3_all = total_dist_3 / num_levels
    
    # Print overall averages
    print(f"Overall Average First Order Coherence: {avg_first_order_coherence_all}")
    print(f"Overall Average Second Order Coherence: {avg_second_order_coherence_all}")
    print(f"Overall Readability: {avg_readability_all}")
    print(f"Overall Diversity dist-2: {avg_dist_2_all}")
    print(f"Overall Diversity dist-3: {avg_dist_3_all}")
    

    return 0

import spacy
#from textdescriptives as td

def average_ppl(examples):
    # Load the spaCy model and add the textdescriptives pipeline component
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("textdescriptives/information_theory")
    
    perplexity_list = []

    # Iterate over each example and calculate perplexity
    count = 0
    for example in examples:
        text =  example["best_response"]
        doc = nlp(text)

        if doc._.perplexity<100:
            #print( example["prompt"])
            #print(text)
            #print(doc._.perplexity)
            #exit()
            # Extract the perplexity for each text
            perplexity_list.append(doc._.perplexity)
    
    # Calculate the average perplexity
    if perplexity_list:
        avg_ppl = sum(perplexity_list) / len(perplexity_list)
    else:
        avg_ppl = None

    return avg_ppl

