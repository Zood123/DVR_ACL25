
from transformers import AutoTokenizer, AutoModelForCausalLM
#import bitsandbytes as bnb
import random
from transformers import BertModel, BertTokenizer
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BitsAndBytesConfig
from utils import String_to_bool,read_examples,save_as_jsonl,verification2bool,save_as_jsonl_dict
import torch
import pickle
import json
from call_tools import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
'''
tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
'''
import sys
import os
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from code.evaluation import run_evaluation
from vllm import LLM, SamplingParams

from tool_set import test_agent, topic_classifier, sentiment_classifier,test_agent_topic_sentiment
import json
#from accelerate import Accelerator
#from gpt_api import GPT_Agent

from instruction_following_eval import default_examples, instruction_following_eval
import torch
from prompt_template import Generate_template_simple, simple_modify_cot,Verify_template,Verify_template_single,simple_modify_topic,simple_modify_fb_topic,Verify_template_topic,Verify_template_single_topic






def Generate_template_simple(prompt):

    few_shots = read_examples("prompt/generation_prompt.jsonl")
    system_prompt = (
        "You are an AI assistant that generates responses based on given prompts. "
        "For each prompt, provide a response that adheres to the specified constraints.\n\n"
    )

    # Initialize the template with the system prompt
    template = f"{system_prompt}\n\n"

    # Append each few-shot example to the template
    for example in few_shots:
        template += f"#Prompt: {example['prompt']}\n\n"
        template += f"Response: {example['output']}\n\n"
    
    # Add the new prompt at the end
    template += f"#Prompt: {prompt}\n\n"
    template += "Response: "

    return template








def Generate_template_topic(prompt):
    few_shots = read_examples(" prompt/generation_prompt_topic.jsonl")
    system_prompt = (
        "You are an AI assistant that generates responses based on given prompts. "
        "For each prompt, provide a response that adheres to the specified constraints.\n\n"
    )

    # Initialize the template with the system prompt
    template = f"{system_prompt}\n\n"

    # Append each few-shot example to the template
    for example in few_shots:
        template += f"#Prompt: {example['instruction']}\n\n"
        template += f"Response: {example['response']}\n\n"
    
    # Add the new prompt at the end
    template += f"#Prompt: {prompt}\n\n"
    template += "Response: "

    return template



def select_verifiers(examples, model,save_path,GT_constriants=False,batch_size=600):
    """
    Generates responses for a list of prompts using a specified model and tokenizer,
    and evaluates the accuracy of the responses based on how well they follow the instructions.
    
    Parameters:
    examples (list): A list of dictionaries, each containing a list of decomposed questions under the key "decomposed questions".
    model: The language model used for generating responses.
    tokenizer: The tokenizer corresponding to the model.
    
    Returns:
    None: Modifies the 'examples' list in-place by adding a 'responses' key with generated responses, and an 'accuracy' key.
    """
    # temp_sentiment
    #batch_size = 600
    input_dicts = []
    count_id = 0
    for i,example in enumerate(examples):

        #examples[i]["sentiment"] = 
        input_dicts.append({"instruction_id":i,"id":count_id,"prompt":temp_topic(example["instruction"]), "arg": None, "verifier": "topic" ,"constraint": "The topic of the instruction.", "valid": False, 'select_item': "topic" ,"budget":5})
        count_id = count_id+1
        input_dicts.append({"instruction_id":i,"id":count_id,"prompt":temp_sentiment(example["instruction"]), "arg": None, "verifier": "sentiment" ,"constraint": "The sentiment of the instruction.","valid": False, 'select_item': "sentiment" ,"budget":5})
        count_id = count_id+1

        


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
                    verify_arg = category_to_parse_senti_topic[batch_input[i]['select_item']](response)
                    if verify_arg != "ERROR":
                        input_dicts[batch_input[i]["id"]]["valid"] = True
                        
                        input_dicts[batch_input[i]["id"]]["arg"] = verify_arg

                    else:
                        print("ERROR!")
                        print(response)
                        if batch_input[i]["budget"] >0:
                            remaining_batch.append(batch_input[i])
                except:
                    print("not valid!")
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
        examples[i]["id"] = i
        if len(example["constraint list"]) != len(example['instruction_id_list']):
            print("mismatch error!")
            exit()
        
        original_prompt = Generate_template_topic(example["instruction"]) #Generate_template_simple(example["prompt"]) #topic_senti_template(example["instruction"]) #Generate_template_simple(example["prompt"]) # Generate_template(example["prompt"])
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
    topic_c = topic_classifier(None)
    sentiment_c = sentiment_classifier(None)
    if max_iteration == 0:
        for i,example in enumerate(examples):
            examples[i]["response"]=all_initial_responses[i]
        
        save_as_jsonl(examples,save_path)
        exit()

    

    for i,example in enumerate(examples):

        #examples[i]["id"] = i
        examples[i]["response"]=all_initial_responses[i]
        examples[i]["initial_response"]=all_initial_responses[i]
        examples[i]["best_response"]=all_initial_responses[i]

        best_score =-1
        example_test = examples[i]
        if len(example_test["instruction_id_list"]) == 0:
            processed_indexes.append(i)
            examples[i]["unsatisfied num"] = 0
            examples[i]["best unsatisfied num"] = 0
            continue
        
        follow_list = test_agent_topic_sentiment(example_test,topic_c,sentiment_c)
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
        #exit()
        unsatisfied_constraints_index = [index for index, is_satisfied in enumerate(example['verify_bool']) if not is_satisfied]
        unsatisfied_constraints = [constraint for constraint, is_satisfied in zip(example["constraint list"], example['verify_bool']) if not is_satisfied]
        unsatisfied_constraints_id = [constraint for constraint, is_satisfied in zip(example_test['instruction_id_list'], example['verify_bool']) if not is_satisfied]
        examples[i]["unsatisfied num"] = len(unsatisfied_constraints)
        examples[i]["best unsatisfied num"] = len(unsatisfied_constraints)
        examples[i]["current_constraint_index"] = unsatisfied_constraints_index[0]
        examples[i]["budget"] = max_iteration

        # note here: We should let the "prompt" be all satisfied except the target constraint:
        # if there are 5 constraints in total, 3 good 2 bad, then the first round prompt should be only 4 constraints and then target the 5th if the 4th is done. Or 4th failed, then just the 5th.
    
    
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
                    if len(additional_shots) > 5:
                        additional_shots = random.sample(additional_shots, 5) # random sample 5
                else:
                    additional_shots = []
                
                # used for non-bank
                if bank == False:
                    additional_shots = []
                if COT:
                    modify_prompt = simple_modify_cot(current_e['instruction'],current_e["best_response"],unsatisfied_constraint,additional_shots) #[]
                else:
                    modify_prompt = simple_modify_fb_topic(current_e['instruction'],current_e["best_response"],unsatisfied_constraint,feedback,additional_shots) #[]
                batch_prompts.append(modify_prompt)
            
            sampling_params = SamplingParams(max_tokens=400,temperature=0.8, top_p=0.95)#,temperature=0.7, top_p=0.95)  #,temperature=0.8, top_p=0.95)
            batch_outputs = model.generate(batch_prompts, sampling_params=sampling_params)

            for k,output in enumerate(batch_outputs):
                response = output.outputs[0].text
                response =response.split('#')
                if COT:
                    cot_process = response[0]
                    try:
                        response = response[1]
                    except:
                        continue
                    prefix = "Modified Response:"
                    if response.startswith(prefix):
                        print("response")
                        print(example["best_response"])
                        print("cot---")
                        print(cot_process)
                        print("response---")
                        batch_list[k]["budget"]=batch_list[k]["budget"]-1
                        response = response[len(prefix):].split('Note:')[0].strip('\n\n\n\n').rstrip("\n")
                        print(response)
                        #exit()
                    else:
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
                example_test["instruction"] = batch_list[k]["instruction"]


                follow_list = test_agent_topic_sentiment(example_test,topic_c,sentiment_c)
                
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
                            correction_memorybank[batch_list[k]['instruction_id_list'][batch_list[k]["current_constraint_index"]]].append({"instruction":batch_list[k]['instruction'],"output":previous_best,"unsatisfied_constraint":batch_list[k]["constraint list"][batch_list[k]["current_constraint_index"]],"chain_of_thought":cot_process,"modified_output":response,"feed_back":previous_feedback[batch_list[k]["current_constraint_index"]]})
                        else:
                            correction_memorybank[batch_list[k]['instruction_id_list'][batch_list[k]["current_constraint_index"]]].append({"instruction":batch_list[k]['instruction'],"output":previous_best,"unsatisfied_constraint":batch_list[k]["constraint list"][batch_list[k]["current_constraint_index"]],"modified_output":response,"feed_back":previous_feedback[batch_list[k]["current_constraint_index"]]})
                        if save_bank_path != None:
                            save_as_jsonl_dict(correction_memorybank,save_bank_path)
                        #break
                
            # next round!
            remaining_batch_list  = []
            for k,example_e in enumerate(batch_list):
                if batch_list[k]["best unsatisfied num"]  == 0 or batch_list[k]["budget"]==0:
                    clean_example = batch_list[k].copy()
                    #try:
                    #    clean_example.pop("unsatisfied num",None)
                    #    clean_example.pop("budget",None)
                    final_list[clean_example['id']] = clean_example
                    
                    #batch_list.pop(k)
                else:
                    previous_index = batch_list[k]["current_constraint_index"]
                    
                    #batch_list[k]["verify_bool"] # check the first False and get the index as the new batch_list[k]["current_constraint_index"] but not the same as previous one
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
                            batch_list[k]["current_constraint_index"] =  current_index  # Return the index of the first False found
                            
                    remaining_batch_list.append(batch_list[k])

            batch_list = remaining_batch_list 
        if len(batch_list)==0 and current_example_id==len(examples):
            all_finish = True

    
    save_as_jsonl(final_list,save_path)







def self_modify_LLMverifier(examples,model,save_path="",vCOT=True,COT=True):

    correction_memorybank = {}

    final_list = []
    max_iteration = 5
    batch_size = 400

    # Split examples into batches
    num_batches = len(examples) // batch_size + (1 if len(examples) % batch_size != 0 else 0)
    original_prompts = []


    for i, example in enumerate(examples):
        
        if len(example["constraint list"]) != len(example['instruction_id_list']):
            print("mismatch error!")
            exit()
        
        original_prompt = Generate_template_topic(example["instruction"]) #Generate_template_simple(example["prompt"]) #topic_senti_template(example["instruction"]) #Generate_template_simple(example["prompt"]) # Generate_template(example["prompt"])
        original_prompts.append(original_prompt)



    all_initial_responses = []
    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, len(examples))
        batch_prompts= original_prompts[batch_start:batch_end]
        
        sampling_params = SamplingParams(max_tokens=400,temperature=0.8, top_p=0.95) 
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
        
        if len(example["instruction_id_list"]) == 0:
            #processed_indexes.append(i)
            examples[i]["unsatisfied num"] = 0
            examples[i]["verify_prompt_list"] = []
            final_list[example["id"]] = example
        else:
            examples[i]["verify_prompt_list"] = Verify_template_topic(example,False)
            examples[i]["unsatisfied num"] = len(example["instruction_id_list"])
            
    
    # initial round: We check in batch!!
    # the batch_size is for the number of samples
    # if the model gives non-valid answer, we can randomly assign one!
    

    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, len(examples))
        batch_exps= examples[batch_start:batch_end]
        batch_prompts = []
        for example in batch_exps:
            batch_prompts.extend(example["verify_prompt_list"])
        sampling_params = SamplingParams(max_tokens=10,temperature=0.8, top_p=0.95) 
        batch_outputs = model.generate(batch_prompts, sampling_params=sampling_params)
        batch_outputs_bool = []
        for output in batch_outputs:
            output = output.outputs[0].text
            try:
                
                output = output.split('\n\n')[0]
                if 'True' in output:
                    batch_outputs_bool.append(True)
                    valid = True
                elif 'False' in output:
                    batch_outputs_bool.append(False)
                    valid = True
                else:
                    batch_outputs_bool.append(False)
                    print("not valid:"+output)
            except:
                print("not valid:"+output)
                batch_outputs_bool.append(True)
        #load them back 
        current_count = 0
        for i in range(len(batch_exps)):
            examples[batch_start+i]["verify_bool"] = batch_outputs_bool[current_count:current_count+len(examples[batch_start+i]["verify_prompt_list"])]
            if examples[batch_start+i]["unsatisfied num"] >0:
                examples[batch_start+i]["unsatisfied num"] = len(examples[batch_start+i]["verify_bool"]) - sum(examples[batch_start+i]["verify_bool"])
            current_count = current_count + len(examples[batch_start+i]["verify_prompt_list"])
    
    for i,example in enumerate(examples):
        if examples[i]["unsatisfied num"] >0:
            unsatisfied_constraints_index = [index for index, is_satisfied in enumerate(example['verify_bool']) if not is_satisfied]
            unsatisfied_constraints = [constraint for constraint, is_satisfied in zip(example["constraint list"], example['verify_bool']) if not is_satisfied]
            #unsatisfied_constraints_id = [constraint for constraint, is_satisfied in zip(example_test['instruction_id_list'], example['verify_bool']) if not is_satisfied]
            #examples[i]["unsatisfied num"] = len(unsatisfied_constraints)
            examples[i]["current_constraint_index"] = unsatisfied_constraints_index[0]
            examples[i]["budget"] = max_iteration


    print("Primary Check Complete")
    #exit()
    # up till now, the examples have been all checked.
    # for each round, one batch --> generation, one batch ---> verify
    
    batch_list = []
    all_finish = False
    current_example_id = 0
    while not all_finish:
        if len(batch_list) < batch_size and current_example_id<len(examples):
            print(current_example_id)
            if examples[current_example_id]["unsatisfied num"] >0:
                batch_list.append(examples[current_example_id])
            else:
                final_list[current_example_id] = examples[current_example_id]
            current_example_id = current_example_id +1
        else:
            batch_prompts =[]
            for current_e in batch_list:
                unsatisfied_constraint = current_e["constraint list"][current_e["current_constraint_index"]]
                if current_e['instruction_id_list'][current_e["current_constraint_index"]] in correction_memorybank:
                    additional_shots = correction_memorybank[current_e['instruction_id_list'][current_e["current_constraint_index"]] ]
                    random.shuffle(additional_shots)
                    if len(additional_shots) > 5: 
                        additional_shots = random.sample(additional_shots, 5)# random sample 5
                else:
                    additional_shots = []
                
                # used for non-bank
                additional_shots = []
                if COT:
                    modify_prompt = simple_modify_cot(current_e['instruction'],current_e["best_response"],unsatisfied_constraint,additional_shots) #[]
                else:
                    modify_prompt = simple_modify_topic(current_e['instruction'],current_e["best_response"],unsatisfied_constraint,additional_shots) #[]
                batch_prompts.append(modify_prompt)
            
            sampling_params = SamplingParams(max_tokens=400,temperature=0.8, top_p=0.95) #temperature=0.8, top_p=0.95
            batch_outputs = model.generate(batch_prompts, sampling_params=sampling_params)
            verify_prompts = []
            responses = []
            for k,output in enumerate(batch_outputs):
                response = output.outputs[0].text
                response =response.split('#')
                if COT:
                    cot_process = response[0]
                    try:
                        response = response[1]
                    except:
                        continue
                    prefix = "Modified Response:"
                    if response.startswith(prefix):
                        print("response")
                        print(example["best_response"])
                        print("cot---")
                        print(cot_process)
                        print("response---")
                        batch_list[k]["budget"]=batch_list[k]["budget"]-1
                        response = response[len(prefix):].split('Note:')[0].strip('\n\n\n\n').rstrip("\n")
                        print(response)
                        #exit()
                    else:
                        print("COT error!!!!!")
                        # not trying to fix yet!!!
                        #exit()
                else:
                    batch_list[k]["budget"]=batch_list[k]["budget"]-1
                    response = response[0].split('Note:')[0].strip('\n\n\n\n').rstrip("\n")
                
                responses.append(response)
                
                unsatisfied_constraint = batch_list[k]["constraint list"][batch_list[k]["current_constraint_index"]]
                verify_prompt = Verify_template_single_topic(batch_list[k]["instruction"],response,unsatisfied_constraint)
                verify_prompts.append(verify_prompt)
            
            sampling_params = SamplingParams(max_tokens=10,temperature=0.8, top_p=0.95)#temperature=0.8, top_p=0.95
            verify_outputs = model.generate(verify_prompts, sampling_params=sampling_params)
            batch_outputs_bool = []
            for verify_output in verify_outputs:
                try:
                    verify_output = verify_output.outputs[0].text
                    verify_output = verify_output.split('\n\n')[0]
                    
                    if 'True' in verify_output:
                        batch_outputs_bool.append(True)
                        valid = True
                    elif 'False' in verify_output:
                        batch_outputs_bool.append(False)
                        valid = True
                    else:
                        batch_outputs_bool.append(False)
                        print("not valid:"+verify_output)
                except:
                    print("not valid:"+verify_output)
                    batch_outputs_bool.append(True)
            
            for k,verify_bool in enumerate(batch_outputs_bool):
                if verify_bool is True:
                    batch_list[k]["verify_bool"][batch_list[k]["current_constraint_index"]] = True
                    batch_list[k]["best_response"]=responses[k]
                    batch_list[k]["unsatisfied num"] = batch_list[k]["unsatisfied num"]-1

                else:
                    pass
            
            # next round
            remaining_batch_list  = []
            for k,example_e in enumerate(batch_list):
                if batch_list[k]["unsatisfied num"]  == 0 or batch_list[k]["budget"]==0:
                    clean_example = batch_list[k].copy()
                    final_list[clean_example['id']] = clean_example
                    
                    #batch_list.pop(k)
                else:
                    previous_index = batch_list[k]["current_constraint_index"]
                    
                    #batch_list[k]["verify_bool"] # check the first False and get the index as the new batch_list[k]["current_constraint_index"] but not the same as previous one
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
                            batch_list[k]["current_constraint_index"] =  current_index  # Return the index of the first False found
                            
                    remaining_batch_list.append(batch_list[k])
            batch_list = remaining_batch_list
        if len(batch_list)==0 and current_example_id==len(examples):
            all_finish = True
    
    save_as_jsonl(final_list,save_path)
    exit()





 






def self_modify_rotate(examples,model,save_path="",COT=True,batch_size=1,bank=False):
    
    max_iteration = 5
    correction_memorybank = {}
    topic_c = topic_classifier(None)
    sentiment_c = sentiment_classifier(None)
    # Split examples into batches
    num_batches = len(examples) // batch_size + (1 if len(examples) % batch_size != 0 else 0)
    original_prompts = []
    for i, example in enumerate(examples):
        if len(example["constraint list"]) != len(example['instruction_id_list']):
            print("mismatch error!")
            exit()
        
        original_prompt = Generate_template_topic(example["instruction"]) #Generate_template_simple(example["prompt"]) #topic_senti_template(example["instruction"]) #Generate_template_simple(example["prompt"]) # Generate_template(example["prompt"])
        original_prompts.append(original_prompt)

    
    all_initial_responses = []
    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, len(examples))
        batch_prompts= original_prompts[batch_start:batch_end]
        
        sampling_params = SamplingParams(max_tokens=400,temperature=0.8, top_p=0.95) 
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

        example_test = example.copy()
        if len(example_test["instruction_id_list"]) == 0:
            processed_indexes.append(i)
            examples[i]["unsatisfied num"] = 0
            #final_list[example["id"]] = example
            continue
        

        
        follow_list = test_agent_topic_sentiment(example_test,topic_c,sentiment_c)
        verify_bool= []
        for element in follow_list:
            if element is  True:
                verify_bool.append(True)
            else:
                verify_bool.append(False)
        unsatisfied_num  = len(verify_bool) - sum(verify_bool)
            
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
            examples[i]["unsatisfied num"] = 0
            #final_list[example["id"]] = example
            continue

        unsatisfied_constraints_index = [index for index, is_satisfied in enumerate(example['verify_bool']) if not is_satisfied]
        unsatisfied_constraints = [constraint for constraint, is_satisfied in zip(example["constraint list"], example['verify_bool']) if not is_satisfied]
        unsatisfied_constraints_id = [constraint for constraint, is_satisfied in zip(example_test['instruction_id_list'], example['verify_bool']) if not is_satisfied]
        examples[i]["unsatisfied num"] = len(unsatisfied_constraints)
        
        examples[i]["current_constraint_index"] = unsatisfied_constraints_index[0]
        examples[i]["budget"] = max_iteration

        # note here: We should let the "prompt" be all satisfied except the target constraint:
        # if there are 5 constraints in total, 3 good 2 bad, then the first round prompt should be only 4 constraints and then target the 5th if the 4th is done. Or 4th failed, then just the 5th.
    
    
    all_finish = False
    batch_list = []
    current_example_id = 0
    while not all_finish:
        if len(batch_list) < batch_size and current_example_id<len(examples):
            print(current_example_id)
            if examples[current_example_id]["unsatisfied num"] >0:
                batch_list.append(examples[current_example_id])
            else:
                final_list[current_example_id] = examples[current_example_id]
            
            current_example_id = current_example_id +1
        else:

            batch_prompts =[]
            for current_e in batch_list:
                unsatisfied_constraint = current_e["constraint list"][current_e["current_constraint_index"]]
                if current_e['instruction_id_list'][current_e["current_constraint_index"]] in correction_memorybank:
                    additional_shots = correction_memorybank[current_e['instruction_id_list'][current_e["current_constraint_index"]] ]
                    random.shuffle(additional_shots)
                    if len(additional_shots) > 5: 
                        additional_shots = random.sample(additional_shots, 5)# random sample 5
                else:
                    additional_shots = []
                
                # used for non-bank
                if bank == False:
                    additional_shots = []
                if COT:
                    modify_prompt = simple_modify_cot(current_e['instruction'],current_e["best_response"],unsatisfied_constraint,additional_shots) #[]
                else:
                    modify_prompt = simple_modify_topic(current_e['instruction'],current_e["best_response"],unsatisfied_constraint,additional_shots) #[]
                batch_prompts.append(modify_prompt)
            
            sampling_params = SamplingParams(max_tokens=400,temperature=0.8, top_p=0.95) #temperature=0.8, top_p=0.95
            batch_outputs = model.generate(batch_prompts, sampling_params=sampling_params)

            for k,output in enumerate(batch_outputs):
                response = output.outputs[0].text
                response =response.split('#')
                if COT:
                    cot_process = response[0]
                    try:
                        response = response[1]
                    except:
                        continue
                    prefix = "Modified Response:"
                    if response.startswith(prefix):
                        print("response")
                        print(example["best_response"])
                        print("cot---")
                        print(cot_process)
                        print("response---")
                        batch_list[k]["budget"]=batch_list[k]["budget"]-1
                        response = response[len(prefix):].split('Note:')[0].strip('\n\n\n\n').rstrip("\n")
                        print(response)
                        #exit()
                    else:
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
                example_test["instruction"] = batch_list[k]["instruction"]

                follow_list = test_agent_topic_sentiment(example_test,topic_c,sentiment_c)
                verify_bool= []
                for element in follow_list:
                    if element is  True:
                        verify_bool.append(True)
                    else:
                        verify_bool.append(False)
                unsatisfied_num  = len(verify_bool) - sum(verify_bool)

                if unsatisfied_num <  batch_list[k]["unsatisfied num"]:

                    previous_best = example["best_response"]
                    batch_list[k]["best_response"]=response
                    batch_list[k]['verify_bool'] = verify_bool
                    batch_list[k]["unsatisfied num"] = unsatisfied_num
                    batch_list[k]["budget"] = max_iteration
                    
                    if verify_bool[batch_list[k]["current_constraint_index"]]:

                        if batch_list[k]['instruction_id_list'][batch_list[k]["current_constraint_index"]] not in correction_memorybank:
                            correction_memorybank[batch_list[k]['instruction_id_list'][batch_list[k]["current_constraint_index"]]] = []
                        if COT:
                            correction_memorybank[batch_list[k]['instruction_id_list'][batch_list[k]["current_constraint_index"]]].append({"instruction":batch_list[k]['instruction'],"output":previous_best,"unsatisfied_constraint":batch_list[k]["constraint list"][batch_list[k]["current_constraint_index"]],"chain_of_thought":cot_process,"modified_output":response})
                        else:
                            correction_memorybank[batch_list[k]['instruction_id_list'][batch_list[k]["current_constraint_index"]]].append({"instruction":batch_list[k]['instruction'],"output":previous_best,"unsatisfied_constraint":batch_list[k]["constraint list"][batch_list[k]["current_constraint_index"]],"modified_output":response})

                
            # next round!
            remaining_batch_list  = []
            for k,example_e in enumerate(batch_list):
                if batch_list[k]["unsatisfied num"]  == 0 or batch_list[k]["budget"]==0:
                    clean_example = batch_list[k].copy()
                    #try:
                    #    clean_example.pop("unsatisfied num",None)
                    #    clean_example.pop("budget",None)
                    final_list[clean_example['id']] = clean_example
                    
                    #batch_list.pop(k)
                else:
                    previous_index = batch_list[k]["current_constraint_index"]
                    
                    #batch_list[k]["verify_bool"] # check the first False and get the index as the new batch_list[k]["current_constraint_index"] but not the same as previous one
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
                            batch_list[k]["current_constraint_index"] =  current_index  # Return the index of the first False found
                            
                    remaining_batch_list.append(batch_list[k])

            batch_list = remaining_batch_list 
        if len(batch_list)==0 and current_example_id==len(examples):
            all_finish = True

    
    save_as_jsonl(final_list,save_path)




if __name__ == "__main__":

    examples= read_examples(" /para_multi_lite_test.jsonl")
 

    #decomposed_constraints = []
    model_name = "mistralai/Mistral-7B-Instruct-v0.3" #"meta-llama/Meta-Llama-3-8B-Instruct" #"hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"  #-Instruct-Float16" #"meta-llama/Meta-Llama-3-8B-Instruct" #"meta-llama/Meta-Llama-3-8B-Instruct" #"lmsys/vicuna-13b-v1.5" #"WizardLMTeam/WizardLM-13B-V1.2" #"meta-llama/Meta-Llama-3-70B-Instruct" #"meta-llama/Meta-Llama-3-8B-Instruct"  # "meta-llama/Llama-2-13b-chat-hf" # meta-llama/Meta-Llama-3-70B-Instruct google/gemma-7b-it "meta-llama/Llama-2-7b-chat-hf"  huggyllama/llama-7b  mistralai/Mistral-7B-Instruct-v0.2  WizardLM/WizardLM-13B-V1.2  mistralai/Mixtral-8x7B-Instruct-v0.1 # Example model name  
    model = LLM(model=model_name,gpu_memory_utilization=0.90)#,tensor_parallel_size=2)#,quantization="AWQ")#,max_model_len=8032) # ,quantization="awq"
    #examples=examples[:20]
    #model = AutoModelForCausalLM.from_pretrained(model_name,token = token,  device_map="auto")#.half()#.to("cuda") , load_in_8bit=True
    #model.save_pretrained("/data/xrzhang/LLM_weights/Meta-Llama-3.1-70B-Instruct-Float16")
    #tokenizer = AutoTokenizer.from_pretrained(model_name,token=token,padding_side='left')

    #generate_verifications(examples=examples,model=model,tokenizer=tokenizer)
    #examples, model =accelerator.prepare(examples, model)
    #generate_responses(examples=examples,model=model,task="Decomposition",batch_size=400,save_path=) #
    #generate_verifications(examples=examples,model=model,save_path= ,GT_constriants=False) 
    #select_verifiers(examples=examples,model=model,save_path= ,GT_constriants=False, batch_size = 600) 
    self_modify_Tfeedback(examples=examples,model=model,save_path=" ",COT=False,batch_size=300,bank=True,warmstart=True, load_bank_path=" ") 
