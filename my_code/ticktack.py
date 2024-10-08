from my_code.utils import convert_to_tensor, multiply_state_dicts, sum_state_dicts, convert_numpy_to_tensor, calculate_new_global_model, apply_softmax_and_replace,calculate_skalar_weighted_model, get_weights, load_or_create_dict
import pandas as pd
import json
import torch
import os

def ticktack(local_tensors,tensor_db,fl_round,collaborators_chosen_each_round):
    #if fl_round == 0:         
    #    return initialize_tick_tack(collaborators_chosen_each_round)        
#
    #el 
    if fl_round % 2 == 0:        
        return tick(collaborators_chosen_each_round=collaborators_chosen_each_round,tensor_db=tensor_db,fl_round=fl_round)
    else:       
        return tack(tensor_db, fl_round, collaborators_chosen_each_round)



def tick(collaborators_chosen_each_round, tensor_db, fl_round):
    root_path = os.path.abspath(os.getcwd())
    path = f"{root_path}/current_weights.json"
    old_weights = load_or_create_dict(path,collaborators_chosen_each_round[0])
    old_weights = {k: v for k, v in old_weights.items() if k in collaborators_chosen_each_round[fl_round]}
    ow_sum = sum(old_weights.values())
    old_weights = {key: value / ow_sum for key, value in old_weights.items()}

    new_models=[]
    # Perform the search to get the entire DataFrame
    for colab in collaborators_chosen_each_round[fl_round]:
        df = tensor_db.search(tags=(f'{colab}','trained'))
        # Step 1: Find the maximum round value
        max_round = df['round'].max()

        # Step 2: Filter the DataFrame to keep only the entries with the highest round
        highest_round_df = df[df['round'] == max_round]

        new_models.append((
            highest_round_df.set_index('tensor_name')['nparray'].to_dict(),
            old_weights[colab]))
    # Extract the relevant information for each collaborator
    
    # Zusammenzählen   
    result = calculate_skalar_weighted_model(new_models)
    torch.save(result, f"{root_path}/state_dict_{fl_round}.h5")
    return result


def tack(tensor_db, fl_round, collaborators_chosen_each_round):
    ### SIEHE constant_hyper_parameters
    learning_rate = 5e-5
    root_path = os.path.abspath(os.getcwd())

    path = f"{root_path}/current_weights.json"
    old_weights = load_or_create_dict(path,collaborators_chosen_each_round)
    new_models=[]
    grads = {}
    prev_models = {}
    for colab in collaborators_chosen_each_round[fl_round]:
      
        previous_tensor_dict = tensor_db.search(fl_round=fl_round-1, tags=(f'{colab}','trained')).set_index('tensor_name')['nparray'].to_dict()
        prev_models[colab] = previous_tensor_dict
        cur_tensor_dict = tensor_db.search(fl_round=fl_round, tags=(f'{colab}','trained')).set_index('tensor_name')['nparray'].to_dict()
            # Gradients = Deltas / Learning Rate # Deltas = local_tensor_dict - global_tensor_dict

        grads[colab] = {k: (cur_tensor_dict[k] - previous_tensor_dict[k])/learning_rate for k in cur_tensor_dict}# if not k.startswith('__opt_')}
        #calcuclate average and derivation
        stats = {k: {'mean': v.mean().item(), 'std': v.std().item()} for k, v in grads[colab].items()}
        # write to file
        with open(f"{root_path}/{fl_round}_{colab}_gradscheck.json", "a") as f:
            json.dump(stats, f)    

    new_weights = get_weights(grads)  
    weights = {k: (old_weights[k] + 5*new_weights[k]) for k in new_weights}
    all_sum = sum(weights.values())
    weights = {key: value / all_sum for key, value in weights.items()}
    json.dump(weights, open(f"{root_path}/weights{fl_round}.json", "w"))
    
    # replace all old_weights with new_weights
    all_weights = old_weights    
    for k in weights:
        all_weights[k] = weights[k]        
    json.dump(all_weights, open(f"{root_path}/current_weights.json", "w"))


    for colab in collaborators_chosen_each_round[fl_round]:   
        new_models.append((prev_models[colab],weights[colab]))
    res = calculate_skalar_weighted_model(new_models)
    torch.save(res, f"{root_path}/state_dict_{fl_round}.h5")
    return res


    


