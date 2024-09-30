from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

#import bitsandbytes as bnb
import torch
from transformers import BertModel, BertTokenizer
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BitsAndBytesConfig
from utils import String_to_bool,read_examples,save_as_jsonl,verification2bool,save_as_jsonl_dict
import torch
import json
import pickle
import json
from call_tools import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


import sys
import os
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from code.evaluation import run_evaluation
from vllm import LLM, SamplingParams

from tool_set import test_agent
import json
 

from instruction_following_eval import default_examples, instruction_following_eval
import torch
from prompt_template import Generate_template_simple, simple_modify_cot,simple_modify,Verify_template,Verify_template_single,simple_modify_fb
 

def Decomposition_simple_template_v2(example):
    role_definition = (
        "You are an advanced assistant specializing in identifying and listing output constraints "
        "from provided instructions. The instructions typically include a task related to generating "
        "content on a specific topic and one (or multiple) format constraint(s). Your goal is to focus only on extracting "
        "and listing all the format constraints required for the output, ignoring the content-related task."
    )


    instruction = example["prompt"]
    constraints_example = (
        "#1. Put your entire answer in JSON format. "
        "#2. The word 'show' should not appear in your response. "
        "#3. Use square brackets for placeholders, like [username1], [username2]. Please include at least 2 placeholders in the thread. "
        "#4. You are not allowed to use any commas in your response. "
    )
    
    #constraints_example = (
    #    "# Put your entire answer in JSON format #,"
    #    "# The word 'weather' should not appear in your response#"
    #)

    example_prompt = (
        "Please generate a few lines of text that touch on the topic of tv. "
        "Put your entire answer in JSON format. The word 'show' should not appear in your response. "
        "Use square brackets for placeholders, like [username1], [username2]. Please include at least 2 placeholders in the thread.You are not allowed to use any commas in your response."
    )

    example_prompt2 = (
        "Can you provide me with some information about dining? "
        "The response must contain a title wrapped in double angular brackets, i.e. <<title>>. "
        "Your answer must contain exactly 4 bullet point in Markdown using the following format:\n* Bullet point one.\n* Bullet point two.\n... "
        "Don't forget to include the keywords her."
    )

    constraints_example2 = (
        "#1. The response must contain a title wrapped in double angular brackets, i.e. <<title>>. "
        "#2. Your answer must contain exactly 4 bullet point in Markdown using the following format:\n* Bullet point one.\n* Bullet point two.\n... "
        "#3. Don't forget to include the keywords her. "
    )

    example_prompt3 = (
    "I need a few texts on the subject of science. Make sure there are exactly 4 sections. Separated the sections by the markdown divider: ***. Make sure the entire response is in English and no capital letters are used. Respond with less than 6 sentences. Do not mention the word Treaty."
    )
    constraints_example3 = (
    "#1. Make sure there are exactly 4 sections. Separated the sections by the markdown divider: ***. "
    "#2. Make sure the entire response is in English and no capital letters are used. "
    "#3. Respond with less than 6 sentences. "
    )

    example_prompt4= (
        "Could you produce some informative text about food or dining? "
        "Put your entire answer in JSON format.")
    constraints_example4 = (
    "#1. Put your entire answer in JSON format. "
    )

 

    template = (
        f"{role_definition}\n\n"
        f"Instruction:\n{example_prompt}\n"
        f"Format Constraints:\n{constraints_example} \n\n"
        f"Instruction:\n{example_prompt3}\n"
        f"Format Constraints:\n{constraints_example3} \n\n"
        f"Instruction:\n{example_prompt2}\n"
        f"Format Constraints:\n{constraints_example2}\n\n"
        f"Instruction:\n{example_prompt4}\n"
        f"Format Constraints:\n{constraints_example4}\n\n"
        f"Instruction:\n{instruction}\n" 
        "Format Constraints:\n#1. "
    )

    return template


 


  
 



def fill_para(examples, model,save_path,GT_constriants=False,batch_size=600):
 
 
    input_dicts = []
    count_id = 0
    for i,example in enumerate(examples): 
        for j,constraint in enumerate(example['constraint list']):
            #print(i)
            #print(j)
            current_match = category_to_match[example['select_list'][j]]
            prompt = category_to_function[example['select_list'][j]](constraint)
            input_dicts.append({"instruction_id":i,"id":count_id,"prompt":prompt, "arg": None, "verifier": current_match ,"constraint": constraint, "valid": False, 'select_item': example['select_list'][j] ,"budget":5}) # 10
            count_id = count_id+1
            if prompt == None:
                input_dicts[-1]['arg'] = {}
                input_dicts[-1]['valid'] = True


    all_finish = False
    batch_input = []
    current_index = 0
    while not all_finish:
        print(current_index)
        if len(batch_input) < batch_size and current_index <len(input_dicts):
            if input_dicts[current_index]["valid"] == False:
                batch_input.append(input_dicts[current_index])
            current_index = current_index+1
        else:
            batch_prompts = []
            for example in batch_input:
                batch_prompts.append(example["prompt"])


            sampling_params = SamplingParams(max_tokens=20,temperature=0.8, top_p=0.95) #temperature=0.8, top_p=0.95
            batch_outputs = model.generate(batch_prompts, sampling_params=sampling_params)

            remaining_batch = []
            for i,output in enumerate(batch_outputs):
                response = output.outputs[0].text.split("\n\n")[0].strip()#.strip("(").strip(")")
                batch_input[i]["budget"] = batch_input[i]["budget"] -1
                try:
                    verify_arg = category_to_parse_function[batch_input[i]['select_item']](response)
                    if verify_arg != "ERROR":
                        input_dicts[batch_input[i]["id"]]["valid"] = True
                        
                        input_dicts[batch_input[i]["id"]]["arg"] = verify_arg

                    else:
                        print("ERROR!")
                        #print(constraint)
                        print(response)
                        if batch_input[i]["budget"] >0:
                            remaining_batch.append(batch_input[i])
                except:
                    print("not valid!")
                    #print(constraint)
                    #print(category_to_match[example['select_list'][j]])
                    print(response)
                    if batch_input[i]["budget"] >0:
                        remaining_batch.append(batch_input[i])
            batch_input=remaining_batch
        if len(batch_input)==0 and current_index==len(input_dicts):
            all_finish=True
                
    args_list = []
    verifiers_list = []
    new_constraints = []
    instruction_id = 0
    
    for i,value in enumerate(input_dicts):
        

        if value['instruction_id'] == instruction_id:
            if value['valid'] == True:
                args_list.append(value['arg'])
                verifiers_list.append(value['verifier'])
                new_constraints.append(value["constraint"])
        else:
            examples[instruction_id]['kwargs']  = args_list
            examples[instruction_id]["instruction_id_list"] = verifiers_list
            examples[instruction_id]['constraint list'] = new_constraints

            while value['instruction_id'] > (instruction_id+1):
                instruction_id = instruction_id +1
                examples[instruction_id]['kwargs']  = []
                examples[instruction_id]["instruction_id_list"] = []
                examples[instruction_id]['constraint list'] = []


            instruction_id = instruction_id +1

            args_list = []
            verifiers_list = []
            new_constraints = []
            if value['valid'] == True:
                args_list.append(value['arg'])
                verifiers_list.append(value['verifier'])
                new_constraints.append(value["constraint"])
    
    examples[instruction_id]['kwargs']  = args_list
    examples[instruction_id]["instruction_id_list"] = verifiers_list
    examples[instruction_id]['constraint list'] = new_constraints
    print(instruction_id)
    save_as_jsonl(examples,save_path)






def prepare_tools(examples, model,save_path,GT_constriants=False):
 
    batch_size =600
    input_dicts = []
    count_id = 0
    for i, example in enumerate(examples):
        prompts = constraint_recog_23full(example["constraint list"])
        for j,prompt in enumerate(prompts):
            input_dicts.append({"instruction_id":i,"id":count_id,"prompt":prompt, "response": None, "constraint":example["constraint list"][j],"valid": False, "budget":5}) #15
            count_id = count_id+1


    all_finish = False
    batch_input = []
    current_index = 0
    while not all_finish:
        print(current_index)
        if len(batch_input) < batch_size and current_index <len(input_dicts):
            batch_input.append(input_dicts[current_index])
            current_index = current_index+1
        else:

            batch_prompts = []
            for example in batch_input:
                batch_prompts.append(example["prompt"])

            sampling_params = SamplingParams(max_tokens=15,temperature=0.8, top_p=0.95) #temperature=0.8, top_p=0.95
            batch_outputs = model.generate(batch_prompts, sampling_params=sampling_params)


            remaining_batch = []
            for i,output in enumerate(batch_outputs):
                response = output.outputs[0].text.split("\n\n")[0].strip().strip("(").strip(")")
                if response not in CATEGORIES:  # Check if the response is valid
                    
                    match_one = find_best_match(response,CATEGORIES)

                    if match_one == None:# or match_one == null:
                        print("--invalid response:---")
                        #print()
                        print(response)
                        batch_input[i]["budget"] = batch_input[i]["budget"] -1
                        if batch_input[i]["budget"] >0:
                            remaining_batch.append(batch_input[i])
                    
                    else:
                        input_dicts[batch_input[i]["id"]]["valid"] = True
                        input_dicts[batch_input[i]["id"]]["response"] = match_one

                else:
                    input_dicts[batch_input[i]["id"]]["valid"] = True
                    input_dicts[batch_input[i]["id"]]["response"] = response
                
            batch_input=remaining_batch
        
        if len(batch_input)==0 and current_index==len(input_dicts):
            all_finish=True
    

    responses = []
    new_constraints = []
    instruction_id = 0
    for i,value in enumerate(input_dicts):
        if value['instruction_id'] == instruction_id:
            if value['valid'] == True:
                responses.append(value['response'])
                new_constraints.append(value["constraint"])
        else:
            examples[instruction_id]['select_list']  = responses
            examples[instruction_id]['constraint list'] = new_constraints

            instruction_id = instruction_id +1
            # instruction_id+1
            responses=[]
            new_constraints = []
            
            if value['valid'] == True:
                responses.append(value['response'])
                new_constraints.append(value["constraint"])
    # for the last instruction
    examples[instruction_id]['select_list']  = responses
    examples[instruction_id]['constraint list'] = new_constraints
    
    print(instruction_id)
    save_as_jsonl(examples,save_path)

    return examples

 



def generate_responses(examples, model,task = None, batch_size=128,save_path = ""):
 
    # Split examples into batches
    num_batches = len(examples) // batch_size + (1 if len(examples) % batch_size != 0 else 0)
    original_prompts = []
    for i, example in enumerate(examples):
        if task == "Decomposition":
            original_prompt = Decomposition_simple_template_v2(example) #Decomposition_simple_template(example)
        elif task == "Verification":
            original_prompt = Verify_template(example)
        elif task == "Generation":
            original_prompt = Generate_template_simple(example["prompt"])
        
        original_prompts.append(original_prompt)

    
    all_initial_responses = []
    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, len(examples))
        batch_prompts= original_prompts[batch_start:batch_end]
        
        sampling_params = SamplingParams(max_tokens=200,temperature=0.8, top_p=0.95)  
        batch_outputs = model.generate(batch_prompts, sampling_params=sampling_params)
        for output in batch_outputs:
            all_initial_responses.append(output.outputs[0].text) #split('#')[0].split('Note:')[0].strip('\n\n\n\n').rstrip("\n")
    

    for i, response in enumerate(all_initial_responses):
        if task == "Decomposition":
             examples[i]['decomposed constraints'] = response
        elif task == "Verification":
            examples[i]['verify_list'] = response
        elif task == "Generation":
            examples[i]['response'] = response
    
    
    results_list = examples
    save_as_jsonl(results_list, save_path)
    return None
    
 
 
 

import random


def load_bank(load_bank_path):
    loaded_set ={}
    correction_memorybank_load =  read_examples(load_bank_path)
    for constraint in correction_memorybank_load:
        #count = 0
        for key, value in constraint.items():
            #count = count +1
            loaded_set[key] = value
        #print(count)
    #exit()
    return loaded_set






def self_modify_Tfeedback(examples,model,save_path="",COT=True,batch_size=1,bank=False,warmstart=False,save_bank_path=None,load_bank_path=None):
    
    max_iteration = 5

    if warmstart:
        correction_memorybank =  load_bank(load_bank_path)
    else:
        correction_memorybank = {}


    # Split examples into batches
    num_batches = len(examples) // batch_size + (1 if len(examples) % batch_size != 0 else 0)
    original_prompts = []
    for i, example in enumerate(examples):
        if len(example["constraint list"]) != len(example['instruction_id_list']):
            print("mismatch error!")
            exit()
        
        original_prompt = Generate_template_simple(example["prompt"]) #Generate_template_simple(example["prompt"]) #topic_senti_template(example["instruction"]) #Generate_template_simple(example["prompt"]) # Generate_template(example["prompt"])
        original_prompts.append(original_prompt)

    
    all_initial_responses = []
    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, len(examples))
        batch_prompts= original_prompts[batch_start:batch_end]
        
        sampling_params = SamplingParams(max_tokens=400,temperature=0.8, top_p=0.95)#,temperature=0.8, top_p=0.95) #) 
        batch_outputs = model.generate(batch_prompts, sampling_params=sampling_params)
        for output in batch_outputs:
            all_initial_responses.append(output.outputs[0].text.split('#')[0].split('Note:')[0].strip('\n\n\n\n').rstrip("\n"))
    
    
    
    final_list = examples.copy()
    processed_indexes = []
    

    

    for i,example in enumerate(examples):
        #print(i)
        examples[i]["id"] = i
        examples[i]["response"]=all_initial_responses[i]
        examples[i]["initial_response"]=all_initial_responses[i]
        examples[i]["best_response"]=all_initial_responses[i]
        #example["best_score"] = 0
        #print(response)
        #exit()
        #generation =initial_response
        best_score =-1

        try:
            example_test = examples[i].copy()
            example_test.pop('initial_response',None)
            example_test.pop('id',None)
            example_test.pop('follow_instruction_list',None)
            example_test.pop('decomposed constraints',None)
            example_test.pop('follow_all_instructions',None)
            example_test.pop('constraint list',None)
            example_test.pop('verify_list',None)
            example_test.pop('verify_bool',None)
            example_test.pop('constraints',None)
            example_test.pop('best_response',None)
            example_test.pop('select_list', None)
            example_test["key"] = 0
            #test_examples.append(example_test)
        except:
            print("fail")
            pass

        if len(example_test["instruction_id_list"]) == 0:
            processed_indexes.append(i)
            examples[i]["unsatisfied num"] = 0
            examples[i]["best unsatisfied num"] = 0
            #final_list[example["id"]] = example
            continue
        
        follow_list = test_agent(example_test)
        examples[i]['feed_back'] = follow_list

        verify_bool = []
        for element in follow_list:
            if element == True:
                verify_bool.append(True)
            else: 
                verify_bool.append(False)

            
        true_ratio = sum(verify_bool) / len(verify_bool)
        print(examples[i]["response"])
        print(true_ratio)
        examples[i]['verify_bool'] = verify_bool
        best_score = true_ratio
        if true_ratio==1.0:
            print("-----best one -------")
            print(best_score)
            print(examples[i]["best_response"])
            processed_indexes.append(i)
            examples[i]["best unsatisfied num"] = 0
            examples[i]["unsatisfied num"] = 0
            #final_list[example["id"]] = example
            continue
        unsatisfied_constraints_index = [index for index, is_satisfied in enumerate(example['verify_bool']) if not is_satisfied]
        unsatisfied_constraints = [constraint for constraint, is_satisfied in zip(example["constraint list"], example['verify_bool']) if not is_satisfied]
        unsatisfied_constraints_id = [constraint for constraint, is_satisfied in zip(example_test['instruction_id_list'], example['verify_bool']) if not is_satisfied]
        examples[i]["unsatisfied num"] = len(unsatisfied_constraints)
        examples[i]["best unsatisfied num"] = len(unsatisfied_constraints)
        examples[i]["current_constraint_index"] = unsatisfied_constraints_index[0]
        examples[i]["budget"] = max_iteration

      
    
    all_finish = False
    batch_list = []
    current_example_id = 0
    while not all_finish:
        if len(batch_list) < batch_size and current_example_id<len(examples):
            print(current_example_id)
            if examples[current_example_id]["best unsatisfied num"] >0:
                batch_list.append(examples[current_example_id])
            else:
                final_list[current_example_id] = examples[current_example_id]

            current_example_id = current_example_id +1
        else:

            batch_prompts =[]
            for current_e in batch_list:
                unsatisfied_constraint = current_e["constraint list"][current_e["current_constraint_index"]]
                feedback = current_e["feed_back"][current_e["current_constraint_index"]]
                if current_e['instruction_id_list'][current_e["current_constraint_index"]] in correction_memorybank:
                    additional_shots = correction_memorybank[current_e['instruction_id_list'][current_e["current_constraint_index"]] ]
                    random.shuffle(additional_shots)
 
                else:
                    additional_shots = []
                
                # used for non-bank
                if bank == False:
                    additional_shots = []
                if COT:
                    modify_prompt = simple_modify_cot(current_e['prompt'],current_e["best_response"],unsatisfied_constraint,additional_shots) #[]
                else:
                    modify_prompt = simple_modify_fb(current_e['prompt'],current_e["best_response"],unsatisfied_constraint,feedback,additional_shots) #[]
                batch_prompts.append(modify_prompt)
            
            sampling_params = SamplingParams(max_tokens=400,temperature=0.8, top_p=0.95) 
            batch_outputs = model.generate(batch_prompts, sampling_params=sampling_params)

            for k,output in enumerate(batch_outputs):
                response = output.outputs[0].text
                response =response.split('#')
                if COT:
                    cot_process = response[0]
                    if len(response)>1:
                        response = response[1]
                    else:
                        response = ""
                        #pass
                    prefix = "Modified Response:"
                    if response.startswith(prefix):
                        print("unsatisfied constraint----------")
                        print(batch_list[k]["constraint list"][batch_list[k]["current_constraint_index"]])
                        print("feedback -----------true")
                        print(batch_list[k]["feed_back"][batch_list[k]["current_constraint_index"]])
                        print("response---")
                        print(batch_list[k]["best_response"])
                        print("cot---")
                        print(cot_process)
                        #print("response---")
                        batch_list[k]["budget"]=batch_list[k]["budget"]-1

                        batch_list[k]["cot_process"] = cot_process
                        response = response[len(prefix):].split('Note:')[0].strip('\n\n\n\n').rstrip("\n")
                        print(response)
                        #exit()
                    else:
                        batch_list[k]["cot_process"] = ""
                        print("COT error!!!!!")
                        # not trying to fix yet!!!
                        #exit()
                else:
                    batch_list[k]["budget"]=batch_list[k]["budget"]-1
                    response = response[0].split('Note:')[0].strip('\n\n\n\n').rstrip("\n")
                
                
                example_test = {}
                example_test["key"] = 0
                example_test['instruction_id_list'] = batch_list[k]['instruction_id_list']
                example_test["kwargs"] = batch_list[k]["kwargs"]
                example_test["response"] = response
                example_test["prompt"] = batch_list[k]["prompt"]


                follow_list = test_agent(example_test)
                
                verify_bool= []
                for element in follow_list:
                    if element is  True:
                        verify_bool.append(True)
                    else:
                        verify_bool.append(False)
                unsatisfied_num  = len(verify_bool) - sum(verify_bool)

                
                if unsatisfied_num <  batch_list[k]["best unsatisfied num"]:
                    previous_best = batch_list[k]["best_response"]#.copy()
                    previous_feedback = batch_list[k]['feed_back']#.copy()

                    batch_list[k]['response']= response
                    batch_list[k]['verify_bool'] = verify_bool
                    batch_list[k]["unsatisfied num"] = unsatisfied_num
                    batch_list[k]['feed_back'] = follow_list
                    
                    batch_list[k]["best_response"]=response
                    batch_list[k]["best unsatisfied num"] = unsatisfied_num                   
                    batch_list[k]["budget"] = max_iteration
                    
                    if verify_bool[batch_list[k]["current_constraint_index"]]:

                        if batch_list[k]['instruction_id_list'][batch_list[k]["current_constraint_index"]] not in correction_memorybank:
                            correction_memorybank[batch_list[k]['instruction_id_list'][batch_list[k]["current_constraint_index"]]] = []
                        if COT: 
                            correction_memorybank[batch_list[k]['instruction_id_list'][batch_list[k]["current_constraint_index"]]].append({"prompt":batch_list[k]['prompt'],"output":previous_best,"unsatisfied_constraint":batch_list[k]["constraint list"][batch_list[k]["current_constraint_index"]],"modified_output":response,"feed_back":batch_list[k]['cot_process']})
                        else:
                            correction_memorybank[batch_list[k]['instruction_id_list'][batch_list[k]["current_constraint_index"]]].append({"prompt":batch_list[k]['prompt'],"output":previous_best,"unsatisfied_constraint":batch_list[k]["constraint list"][batch_list[k]["current_constraint_index"]],"modified_output":response,"feed_back":previous_feedback[batch_list[k]["current_constraint_index"]]})
                        if save_bank_path != None:
                            save_as_jsonl_dict(correction_memorybank,save_bank_path)
                        #break
                
            # next round!
            remaining_batch_list  = []
            for k,example_e in enumerate(batch_list):
                if batch_list[k]["best unsatisfied num"]  == 0 or batch_list[k]["budget"]==0:
                    clean_example = batch_list[k].copy()
 
                    final_list[clean_example['id']] = clean_example
                    
                    #batch_list.pop(k)
                else:
                    previous_index = batch_list[k]["current_constraint_index"]
                    
                    verify_bool_list = batch_list[k]["verify_bool"]
                    if batch_list[k]["unsatisfied num"] == 1:
                        batch_list[k]["current_constraint_index"] = verify_bool_list.index(False)
                    # Start looking for False from the index right after the previous_index
                    start_index = previous_index + 1 if previous_index < len(verify_bool_list) - 1 else 0

                    # Iterate over the list, wrap around if necessary
                    for i in range(len(verify_bool_list)):
                    # Calculate current index by wrapping around the list
                        current_index = (start_index + i) % len(verify_bool_list)
                        if verify_bool_list[current_index] == False:
                            batch_list[k]["current_constraint_index"] =  current_index   
                            
                    remaining_batch_list.append(batch_list[k])

            batch_list = remaining_batch_list 
        if len(batch_list)==0 and current_example_id==len(examples):
            all_finish = True

    
    save_as_jsonl(final_list,save_path)


 

import difflib

def find_best_match(constraint,matchset):
    closest_match = difflib.get_close_matches(constraint, matchset, n=1)
    if closest_match:
        return closest_match[0]
    else:
        return None


SELECTIONS = [
    "postscript",
    "placeholder",                       #"detectable_content:number_placeholders"
    "include keyword",
    "letter frequency",
    "keyword frequency",
    "forbidden words",
    "number sentences",
    "number words",
    "separated with ***",
    "bullet lists",
    "constrained_response",
    "mark sections",                               #multiple_sections
    "highlighted sections",
    "json format",
    "title",
    "two responses",
    "quotation",
    "end with",
    "no comma",
    "Only capital",
    "lowercase",
    "Capital word frequency",
    "language"
]




SELECTIONS1 = [
    #"postscript",
    #"placeholder", #"detectable_content:number_placeholders"
    "keyword",
    "length",
    #"format",
    "case change",
    #"language"
]




 



selections = [
    "postscript",
    "placeholder",
    "include keyword",  # existence
    "letter frequency",
    "keyword frequency",
    "forbidden word",
    "sentence count",  # length_constraints:number_sentences
    "word count",  # length_constraints:number_words
    "separators ***",  # length_constraints:number_paragraphs
    "bullet points",
    "fixed responses",
    "marked sections",  # multiple_sections number_sections
    "highlighted sections",
    "json format",
    "title format",
    "two responses",
    "quotation",
    "end phrase",
    "no comma",
    "all caps",
    "all lowercase",
    "capital word frequency",
    "language"
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



CATEGORIES = [
    "postscript",
    "placeholder",
    "include keyword",
    "letter frequency",
    "keyword frequency",
    "exclude keyword",
    "sentence count constraint",
    "word count constraint",
    "*** separator",
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


def constraint_recog_23full(constraints):
    # List of constraint and selection pairs
    constraints_list = [
        ("End it with a post script starting with P.S.", "postscript"),
        ("The response must contain at least 1 placeholder (i.e., [restaurant]).", "placeholder"),
        ("Make sure to include the word 'mutations'.", "include keyword"),
        ("Don't forget to include the keywords her.", "include keyword"),
        ("Provide an answer without using the word 'currency'.", "exclude keyword"),
        ("Make sure the word 'before' appears less than 3 times.", "keyword frequency"),
        ("Ensure the letter 'l' appears less than 8 times in your response.", "letter frequency"),
        ("The number of sentences in your response should be at least 5.", "sentence count constraint"),
        ("Organize your entire response in less than 4 sentences.", "sentence count constraint"),
        ("Limit the number of words you use to fewer than 65 words.", "word count constraint"),
        ("Separate your response into 3 sections, where each section is separated with ***.", "*** separator"),
        ("Your answer must be in the form of exactly 2 bullet points with the format:\n* This is bullet point 1\n* This is bullet point 2.", "bullet points"),
        ("Your response should be one of the following: 'My answer is yes.', 'My answer is no.', 'My answer is maybe.'", "fixed responses"), 
        ("Highlight at least 2 sections of your response in markdown such as *highlighted section*.", "highlighted"),
        ("The entire output should be in JSON format.", "JSON format"),
        ("The response must contain a title wrapped in double angular brackets, i.e., <<title>>.", "title format"),
        ("Wrap the entire response in double quotation marks.", "quoted response"),
        ("Finish the response with the exact phrase: 'Hope you agree with me.'", "end phrase"),
        ("Provide a response without using any commas.", "no commas"),
        ("Ensure to use only capital letters in your entire response.", "all capital letters"),
        ("The answer should be in all lowercase letters, without any capitalizations.", "all lowercase"),
        ("Add stress words which are capitalized. Limit these stress words to fewer than 1 time.", "capital word frequency"),
        ("Respond using only the Persian language; no other language is allowed.", "language restriction")
    ]
    
    categories = [
        "postscript", "placeholder", "include keyword", "exclude keyword","letter frequency", "keyword frequency",
         "sentence count constraint", "word count constraint", "*** separator", "bullet points",
        "fixed responses", "highlighted", "JSON format", "title format",
        "quoted response", "end phrase", "no commas", "all capital letters",
        "all lowercase", "capital word frequency", "language restriction"
    ]
    
    questions = []
    
    # System prompt
    system_prompt = ("You will be given a list of constraints. Each constraint belongs to a specific category. "
                     "Your task is to recognize and categorize each constraint. "
                     "Only output the category from the following options:\n\n" + ", ".join(categories) + "\n\n"
                     "Please ensure to categorize each constraint accurately according to its description. "
                     "There is definitely a valid category option for each constraint. You can Here are examples for each type of constraint:")
    
    
    # Template for generating questions
    for constraint in constraints:
        question = system_prompt
        for constraint_example, selection_example in constraints_list:
            question += f"\n\nPrompt: {constraint_example}\nCategory: {selection_example}"
        question += f"\n\nPrompt: {constraint}\nCategory: "
        questions.append(question)

    return questions


 






if __name__ == "__main__": 
    # change path to the output of previous step
    instruction_path = "data/syn_data_constraints_all.jsonl"
    examples= read_examples(instruction_path)

    model_path = "mistralai/Mistral-7B-Instruct-v0.3"  
    model = LLM(model=model_path,gpu_memory_utilization=0.97,tensor_parallel_size=2) 


    # step 1: decompose, or generation
    #generate_responses(examples=examples,model=model,task="Decomposition",batch_size=400,save_path="data/simple_decompose.jsonl") 
    # step 2: select tools
    #prepare_tools(examples=examples,model=model,save_path="data/select_all.jsonl",GT_constriants=False) 
    # step 3: fill parameters
    #fill_para(examples=examples,model=model,save_path="data/para_all.jsonl",GT_constriants=False, batch_size = 600) 
    # step 4: refinement
    self_modify_Tfeedback(examples=examples,model=model,save_path="data/modify_tool_max5_fb.jsonl",COT=False,batch_size=300,bank=True,warmstart=False) 
