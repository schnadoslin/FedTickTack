import copy
import torch
import numpy as np

import os
import json

def normalize_all_weights(path):
    weights = load_or_create_dict(path,None)
    total_sum = sum(weights.values())
    normalized_weights = {key: value / total_sum for key, value in weights.items()}
    with open(path, 'w') as file:
        json.dump(normalized_weights, file)

def load_or_create_dict(path, collabarators):
    # Check if the file exists
    if os.path.exists(path):
        # If the file exists, load the dictionary from the file
        with open(path, 'r') as file:
            data = json.load(file)
    else:
        # If the file does not exist, create an empty dictionary
        data = {collabarator: (1/len(collabarators)) for collabarator in collabarators}
        # Save the empty dictionary to the file
        with open(path, 'w') as file:
            json.dump(data, file)
    return data


def convert_to_tensor(model):#(state_dict):
    #convert a torch model state_dict to a tensor
    tensor = model.state_dict()
    pass


# Funktion zur Konvertierung von Numpy-Arrays in Torch-Tensoren
def convert_numpy_to_tensor(input_dict):
    for key, value in input_dict.items():
        if(key.endswith('num_batches_tracked')):
            continue
        if isinstance(value, np.ndarray):
            input_dict[key] = torch.from_numpy(value)
        else:            
            input_dict[key] = value.to('cpu')
    return input_dict

def multiply_state_dicts(sd1, sd2):
    s1 = convert_numpy_to_tensor(sd1)
    s2 = convert_numpy_to_tensor(sd2)
    result = {}
        
    for key in s1:
        if key.endswith('weight') or key.endswith('bias'):
            if key in s2:            
                result[key] = s1[key] * s2[key]
        else:
            result[key] = s1[key]
    return result

def sum_state_dicts(state_dicts):
    # Check if the list is empty
    if not state_dicts:
        raise ValueError("The list of state_dicts is empty")

    # Initialize the summed state_dict as a deep copy of the first state_dict
    summed_state_dict = copy.deepcopy(state_dicts[0])
    
    # Iterate through the remaining state_dicts and add their values to the summed_state_dict

    for state_dict in state_dicts[1:]:
        for key in state_dict:            
            summed_state_dict[key] += state_dict[key]    
    return summed_state_dict

import torch.nn.functional as F
def apply_softmax_and_replace(input_data, suffixes=['weight', 'bias']):
    # Dictionary to collect all tensors of the same key
    collected_tensors = {}
    tensor_reference = {}

    # Collect tensors from the nested dictionaries in the input data
    for outer_key, inner_dict in input_data.items():
        for key, tensor in inner_dict.items():
            if any(key.endswith(suffix) for suffix in suffixes):
                if key not in collected_tensors:
                    collected_tensors[key] = []
                    tensor_reference[key] = []
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                collected_tensors[key].append(tensor)
                tensor_reference[key].append((inner_dict, key))  # Keep track of where the tensor came from
    
    # Apply softmax on collected tensors
    softmaxed_tensors = {}
    for key, tensors in collected_tensors.items():
        concatenated_tensors = torch.stack(tensors)
        # Normalize the tensors for numerical stability before applying softmax
        max_vals, _ = torch.max(concatenated_tensors, dim=1, keepdim=True)
        stabilized_tensors = concatenated_tensors - max_vals
        softmaxed_tensor = F.softmax(stabilized_tensors, dim=0)
        softmaxed_tensors[key] = softmaxed_tensor
    
    # Replace original tensors with softmaxed tensors
    for key, tensor_list in tensor_reference.items():
        for idx, (d, key) in enumerate(tensor_list):
            # Replace the original tensor with the softmaxed tensor
            d[key] = softmaxed_tensors[key][idx]
    
    return input_data

def get_weights(input_data, suffixes=['weight', 'bias']):
    meanG = {}
    all_means = []
    
    for outer_key, inner_dict in input_data.items():
        d = []
        for key, tensor in inner_dict.items():
            if any(key.endswith(suffix) for suffix in suffixes):
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                d.append(tensor.abs().mean().item())
        meanG[outer_key] = np.mean(d)
        all_means.append(meanG[outer_key])
    
    print(meanG)
    total_sum = sum(all_means)
    
    if total_sum == 0:
        normalized_meanG = {key: 1.0 / len(meanG) for key in meanG}  # Avoid division by zero if all means are zero
    else:
        # Normalize such that the sum of the normalized means is 1
        normalized_meanG = {key: value / total_sum for key, value in meanG.items()}
    #    normalized_meanG = {key: 1 / value for key, value in meanG.items()}
    #    Gsum = sum(normalized_meanG.values())
    #    normalized_meanG = {key: value / Gsum for key, value in normalized_meanG.items()}

    
    return normalized_meanG  


def calculate_skalar_weighted_model(WandS):
    weighted_models = [{key: value * weight for key, value in model.items()} for model, weight in WandS]
    summed = sum_state_dicts(weighted_models)
    return summed

def calculate_new_global_model(WandS):
    #Führen Sie die Multiplikation der Ursprungsmatrix mit der Gewichtungsmatrix für jedes Tupel durch
    #Summieren Sie die Ergebnisse der Multiplikationen, um eine globale Matrix zu erhalten
    #Normalisieren Sie die globale Matrix positionsweise. Dazu müssen Sie für jede Position in der Matrix den Wert durch die Summe der Gewichte an dieser Position teilen. Dies erfordert, dass Sie während der Multiplikation oder in einem separaten Schritt die Summe der Gewichte für jede Position speichern.
        
    
    
    u = []
    #weights_sum = None
    for i in WandS:
        multiplied = multiply_state_dicts(i[0], i[1])
        u.append(multiplied)
    #    if weights_sum is None:
    #        weights_sum = copy.deepcopy(i[1])  # Initialisieren mit den ersten Gewichten
    #    else:
    #        for key in i[1]:
    #            if key in weights_sum:
    #                weights_sum[key] += i[1][key]     
    #            else:
    #                weights_sum[key] = i[1][key]
                
    global_model = sum_state_dicts(u)
            
    # Normalisierung
    for key in global_model:
#        if key not in weights_sum:
#        global_model[key]= global_model[key] / len(WandS)
        if key.endswith('weight') or key.endswith('bias'):
            continue
            #global_model[key] = global_model[key] / (weights_sum[key])
        else:
            global_model[key] = global_model[key] / len(WandS)
    
    return global_model
    