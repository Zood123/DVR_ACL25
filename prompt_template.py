from utils import read_examples

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








def simple_modify_cot(prompt,output,unsatisfied_constraint,few_shots_frombank=[]):
    # feed_back
    few_shots = read_examples("prompt/modify_cot_examples_v2_5shots.jsonl")
    
    few_shots.extend(few_shots_frombank)
    try: 
        few_shots = random.sample(few_shots,8)
    except:
        few_shots = few_shots
    #few_shots.extend(few_shots_frombank)

    few_shots_text = ""
    for i, shot in enumerate(few_shots):
        few_shots_text += (
            f"#Prompt: {shot['prompt']}\n"
            f"#Original Response: {shot['output']}\n"
            f"#It does not satisfy the constraint: {shot['unsatisfied_constraint']} Let's modify the response.\n"
            f"#Let's think step by step: {shot['feed_back']}\n"
            f"#Modified Response: {shot['modified_output']}\n\n"
        )

    print("----unsatisfied----")
    #print(unsatisfied_constraints)
    #prompt2 = "Describe a peaceful garden. Your description should be at least 4 sentences. Include the word 'serene'. do not use the word 'flower'. Wrap the entire response with double quotation marks."
    #output2 = "\"The garden has many flowers and trees. A small pond lies in the center. Birds are chirping.\""
    #unsatisfied_constraints2 = ['Include the word "serene"', 'Do not use the word "flower"']
    #modified_output2 = "\"The garden has various plants and trees. A small pond lies in the center. Birds are chirping. It's a serene place to relax.\""


    system_prompt = (
    "You are an assistant responsible for refining text generation outputs to ensure they comply with specified constraints. Given a prompt, its original response, and an unsatisfied constraint, your task is to modify the response to fully meet the constraint.\n\n"
    )
    #system_prompt = (
    #    "You are an assistant that helps improve text generation by ensuring they meet all specified constraints. Given a prompt, output, and an unsatisfied constraint"
    #    ", your task is to modify the output to satisfy the unsatisfied constraint.\n\n"
    #)

    prompt_template = (
        f"{system_prompt}"
        f"{few_shots_text}"
        f"#Prompt: {prompt}\n"
        f"#Original Response: {output}\n"
        f"#It does not satisfy the constraint: {unsatisfied_constraint} Let's modify the response.\n"
        "#Let's think step by step: "
    )

    return prompt_template



# this one is to modify the output without COT
def simple_modify(prompt,output,unsatisfied_constraint,few_shots_frombank=[]):

    few_shots = read_examples("prompt/modify_cot_examples_v2_5shots.jsonl")
    

    few_shots.extend(few_shots_frombank)

    few_shots_text = ""
    for i, shot in enumerate(few_shots):
        few_shots_text += (
            f"#Prompt: {shot['prompt']}\n"
            f"#Original Response: {shot['output']}\n"
            f"#It does not satisfy the constraint: {shot['unsatisfied_constraint']} Let's modify the response.\n"
            f"#Modified Response: {shot['modified_output']}\n\n"
        )

    print("----unsatisfied----")
    system_prompt = (
    "You are an assistant responsible for refining text generation outputs. Given a prompt, its original response, and an unsatisfied constraint, your task is to modify the response to fully meet the unsatisfied constraint while maintaining other existing constraints in the prompt.\n\n"
    )
    
    #system_prompt = (
    #    "You are an assistant that helps improve text generation by ensuring they meet all specified constraints. Given a prompt, the original response, and an unsatisfied constraint"
    #    ", your task is to modify the response to satisfy the unsatisfied constraint.\n\n"
    #)

    prompt_template = (
        f"{system_prompt}"
        f"{few_shots_text}"
        f"#Prompt: {prompt}\n"
        f"#Original Response: {output}\n"
        f"#It does not satisfy the constraint: {unsatisfied_constraint} Let's modify the response.\n"
        "#Modified Response: "
    )

    return prompt_template


# this one is to modify the output without COT
def simple_modify_topic(prompt,output,unsatisfied_constraint,few_shots_frombank=[]):

    few_shots = read_examples("prompt/modify_topic_sentiment_shots.jsonl")

    few_shots.extend(few_shots_frombank)

    few_shots_text = ""
    for i, shot in enumerate(few_shots):
        few_shots_text += (
            f"#Instruction: {shot['instruction']}\n"
            f"#Original Response: {shot['output']}\n"
            f"#It does not satisfy the constraint: {shot['unsatisfied_constraint']} Let's modify the response.\n"
            f"#Modified Response: {shot['modified_output']}\n\n"
        )

    print("----unsatisfied----")
    system_prompt = (
    "You are an assistant responsible for refining text generation outputs. Given the instruction, its original response, and an unsatisfied constraint, your task is to modify the response to fully meet the unsatisfied constraint while maintaining other existing constraints in the prompt.\n\n"
    )
    
    #system_prompt = (
    #    "You are an assistant that helps improve text generation by ensuring they meet all specified constraints. Given a prompt, the original response, and an unsatisfied constraint"
    #    ", your task is to modify the response to satisfy the unsatisfied constraint.\n\n"
    #)

    prompt_template = (
        f"{system_prompt}"
        f"{few_shots_text}"
        f"#Instruction: {prompt}\n"
        f"#Original Response: {output}\n"
        f"#It does not satisfy the constraint: {unsatisfied_constraint} Let's modify the response.\n"
        "#Modified Response: "
    )

    return prompt_template





import random
# this one is to modify the output without COT
def simple_modify_fb(prompt,output,unsatisfied_constraint,feedback,few_shots_frombank=[]):

    few_shots = read_examples("prompt/modify_cot_examples_v2_5shots.jsonl")
    few_shots.extend(few_shots_frombank)
    
    #if len(few_shots_frombank) >9:
    #    few_shots = few_shots_frombank
    #else:
    #    few_shots.extend(few_shots_frombank)
    try: 
        few_shots = random.sample(few_shots,8)
    except:
        few_shots = few_shots
    #few_shots = []

    few_shots_text = ""
    for i, shot in enumerate(few_shots):
        few_shots_text += (
            f"#Prompt: {shot['prompt']}\n"
            f"#Original Response: {shot['output']}\n"
            f"#It does not satisfy the constraint: {shot['unsatisfied_constraint']}\n"
            f"#Analysis: {shot['feed_back']}\n"
            f"#Modified Response: {shot['modified_output']}\n\n"
        )

    print("----unsatisfied----")
    system_prompt = (
    "You are an AI assistant responsible for refining a given response. Given a prompt, its original response, and the analysis of the response, your task is to modify the response according to the analysis.\n\n"
    )
    
    #system_prompt = (
    #    "You are an assistant that helps improve text generation by ensuring they meet all specified constraints. Given a prompt, the original response, and an unsatisfied constraint"
    #    ", your task is to modify the response to satisfy the unsatisfied constraint.\n\n"
    #)

    prompt_template = (
        f"{system_prompt}"
        f"{few_shots_text}"
        f"#Prompt: {prompt}\n"
        f"#Original Response: {output}\n"
        f"#It does not satisfy the constraint: {unsatisfied_constraint}\n"
        f"#Analysis: {feedback}\n"
        "#Modified Response: "
    )



    #print(prompt_template)
    #exit()
    return prompt_template


# this one is to modify the output without COT
def simple_modify_fb_topic(prompt,output,unsatisfied_constraint,feedback,few_shots_frombank=[]):

    few_shots = read_examples("prompt/modify_topic_sentiment_shots.jsonl")

    few_shots.extend(few_shots_frombank)
    #if len(few_shots_frombank) >2:
    #    few_shots = few_shots_frombank
    #else:
    #    few_shots.extend(few_shots_frombank)
    few_shots = random.sample(few_shots, 2)
    #few_shots = []
    few_shots_text = ""
    for i, shot in enumerate(few_shots):
        few_shots_text += (
            f"#Instruction: {shot['instruction']}\n"
            f"#Original Response: {shot['output']}\n"
            f"#It does not satisfy the constraint: {shot['unsatisfied_constraint']}\n"
            f"#Analysis: {shot['feed_back']}\n"
            f"#Modified Response: {shot['modified_output']}\n\n"
        )

    print("----unsatisfied----")
    system_prompt = (
    "You are an AI assistant responsible for refining a given response. Given the instruction, its original response, and the analysis of the response, your task is to modify the response according to the analysis.\n\n"
    )
    
    #system_prompt = (
    #    "You are an assistant that helps improve text generation by ensuring they meet all specified constraints. Given a prompt, the original response, and an unsatisfied constraint"
    #    ", your task is to modify the response to satisfy the unsatisfied constraint.\n\n"
    #)

    prompt_template = (
        f"{system_prompt}"
        f"{few_shots_text}"
        f"#Instruction: {prompt}\n"
        f"#Original Response: {output}\n"
        f"#It does not satisfy the constraint: {unsatisfied_constraint}\n"
        f"#Analysis: {feedback}\n"
        "#Modified Response: "
    )



    #print(prompt_template)
    #exit()
    return prompt_template




def Merge_temp(prompt,constraint_list,response_list):
    
    few_shots = read_examples("prompt/Branch_solve.jsonl")
    system_prompt = (
        "You are an AI assistant that merges responses into one single response. "
        "Each response satisfies one single constraint .\n"
        "Your task is to merge these responses so that the final response can satisfy all constraints.\n\n"
    )

    # Initialize the template with the system prompt
    template = f"{system_prompt}\n\n"

    # Append each few-shot example to the template
    for example in few_shots:
        template += f"#Original Prompt: {example['prompt']}\n\n"
        full_prompt = example['prompt']
        for i,constraint in enumerate(example['constraints']):
            full_prompt = full_prompt +" " + constraint
            template += "#Prompt: " + example['prompt']+" "+constraint + "\n"
            template += f"#Response: {example['responses'][i]}\n"

        template += f"#Merged Prompt: {full_prompt}\n"
        template += f"#Merged Response: {example['merged response']}\n\n"
            
    # Add the new prompt at the end

    template += f"#Original Prompt: {prompt}\n\n"
    full_prompt = prompt
    for i,constraint in enumerate(constraint_list):
        full_prompt = full_prompt +" " + constraint
        template += "#Prompt: " + prompt+" "+constraint + "\n"
        template += f"#Response: {response_list[i]}\n"

    template += f"#Merged Prompt: {full_prompt}\n"
    template += "#Merged Response: "
    #print(template)
    #exit()
    return template





def Select_temp(prompt,response_list):


    system_prompt = (
    "Given the following responses to the query, please evaluate and select the most consistent response based on the majority consensus.\n"
    "We will list all responses and their index numbers. You should select one response by outputing its index number")

    # Initialize the template with the system prompt
    template = f"{system_prompt}\n\n"


    few_shots = read_examples(" LLM_project1/prompt/U_SC.jsonl")
    
    for example in few_shots:
        template += f"#I have generated the following responses to the instruction: {prompt}\n"
        for  index, response in enumerate(example["responses"]):
            template += f"Response {index}: {response}\n"
        label = str(example["index"])
        template += f"\nThe most consistent response index: {label}\n\n"

    # Add the new prompt at the end

    template += f"#I have generated the following responses to the instruction: {prompt}\n"

    for index, response in enumerate(response_list):
        template += f"Response {index}: {response}\n"

    # Add instructions for selecting the most consistent response
    template += "\nThe most consistent response index: "

    return template



















# this one is to modify the output without COT
def modify_generalfeedback(prompt,output,unsatisfied_constraint,few_shots_frombank=[]):

    few_shots = read_examples("prompt/modify_cot_examples_v2_5shots.jsonl")

    few_shots.extend(few_shots_frombank)

    few_shots_text = ""
    for i, shot in enumerate(few_shots):
        few_shots_text += (
            f"#Prompt: {shot['prompt']}\n"
            f"#Original Response: {shot['output']}\n"
            f"#Issue: The response does not fully satisfy the specified constraints. Please see the modified version below:\n"
            f"#Modified Response: {shot['modified_output']}\n\n"
        )

    print("----unsatisfied----")
    system_prompt = (
        "You are a skilled assistant specializing in refining text generation outputs to ensure they adhere to specified constraints. "
        "Your task is to modify the given response so that it fully satisfies all constraints. You will be provided with the Prompt and the Original Response "
        "Please modify the response to ensure it meets all constraints.\n\n"
    )
    
    #system_prompt = (
    #    "You are an assistant that helps improve text generation by ensuring they meet all specified constraints. Given a prompt, the original response, and an unsatisfied constraint"
    #    ", your task is to modify the response to satisfy the unsatisfied constraint.\n\n"
    #)

    prompt_template = (
        f"{system_prompt}"
        f"{few_shots_text}"
        f"#Prompt: {prompt}\n"
        f"#Original Response: {output}\n"
        f"#It does not satisfy all the constraints. Let's modify the response.\n"
        "#Modified Response: "
    )

    return prompt_template








def Verify_template(example,GT_constriants):


    few_shots = read_examples("prompt/LLM_verify_shots.jsonl")
    system_prompt = "Please evaluate the following text based on the given constraint and return 'True' or 'False'."
    template = ""
    template = template + system_prompt

    for shot in few_shots:
        response = shot["response"]
        constraint = shot["constraints"][0]
        result = shot["results"][0]
        
        template = template+ (f"\n\nText: #{response}# \n\nThe text satisfies the following constraint: '{constraint}'\n'True' or 'False': {str(result).capitalize()}")


    current_response =example["response"]
    
    if GT_constriants:
        current_constraints = example["constraints"]
    else:
        current_constraints = example["constraint list"]


    all_prompts = []
    for current_single_constraint in current_constraints:
        
            
        current_question = (f"\n\nText: #{current_response}# \n\nThe text satisfies the following constraint: '{current_single_constraint}'\n'True' or 'False':")

        all_prompts.append(template+current_question)
    

    return all_prompts



def Verify_template_single(current_response,current_constraint):


    few_shots = read_examples("prompt/LLM_verify_shots.jsonl")
    system_prompt = "Please evaluate the following text based on the given constraint and return 'True' or 'False'."
    template = ""
    template = template + system_prompt

    for shot in few_shots:
        response = shot["response"]
        constraint = shot["constraints"][0]
        result = shot["results"][0]
        
        template = template+ (f"\n\nText: #{response}# \n\nThe text satisfies the following constraint: '{constraint}'\n'True' or 'False': {str(result).capitalize()}")


            
    current_question = (f"\n\nText: #{current_response}# \n\nThe text satisfies the following constraint: '{current_constraint}'\n'True' or 'False':")
    

    return template+current_question



def Verify_template_single_topic(current_instruction,current_response,current_constraint):


    few_shots = read_examples("prompt/LLM_verify_topic_sentiment.jsonl")
    system_prompt = "Please evaluate the following text based on the given constraint and return 'True' or 'False'."
    template = ""
    template = template + system_prompt

    for shot in few_shots:
        instruction = shot["instruction"]
        response = shot["response"]
        constraint = shot["constraints"][0]
        result = shot["results"][0]
        
        template = template+ (f"\n\nInstruction: {instruction} \n\nText: {response} \n\nThe text satisfies '{constraint}'\n'True' or 'False': {str(result).capitalize()}")
            
    current_question = (f"\n\nInstruction: {current_instruction} \n\nText: {current_response} \n\nThe text satisfies the following constraint: '{current_constraint}'\n'True' or 'False':")
    

    return template+current_question



def Verify_template_topic(example,GT_constriants):

    few_shots = read_examples("prompt/LLM_verify_topic_sentiment.jsonl")
    system_prompt = "Please evaluate the following text based on the given constraint and return 'True' or 'False'."
    template = ""
    template = template + system_prompt

    for shot in few_shots:
        instruction = shot["instruction"]
        response = shot["response"]
        constraint = shot["constraints"][0]
        result = shot["results"][0]
        
        template = template+ (f"\n\nInstruction: {instruction} \n\nText: #{response}# \n\nThe text satisfies the following constraint: '{constraint}'\n'True' or 'False': {str(result).capitalize()}")


    current_response =example["response"]
    current_instruction = example["instruction"]
    
    if GT_constriants:
        current_constraints = example["constraints"]
    else:
        current_constraints = example["constraint list"]


    all_prompts = []
    for current_single_constraint in current_constraints:
        
            
        current_question = (f"\n\nInstruction: {current_instruction} \n\nText: #{current_response}# \n\nThe text satisfies the following constraint: '{current_single_constraint}'\n'True' or 'False':")

        all_prompts.append(template+current_question)
    

    return all_prompts










if __name__ == "__main__":
    print(Merge_temp("123",["345"],["678"]))
