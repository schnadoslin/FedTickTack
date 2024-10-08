import os
import json
import numpy as np
import os
import json
from my_code.utils import normalize_all_weights

def get_current_round(fl_round, collaborators, site_metrics):
    # collaborators: list of strings of collaborator names
   root_path = os.path.abspath(os.getcwd())
   round_path = f'{root_path}/rounds.json'
   anzahlLabel=3
   batcher = {}
   for b in range(3,5):
       batcher[b] = len(site_metrics)%(b*anzahlLabel)           
   calc_num_batches = min(batcher, key=batcher.get)
   if fl_round == 0 or fl_round == 1:

        return collaborators

   else:
        if not os.path.exists(round_path):
            with open(round_path, 'w') as f:
                json.dump({}, f)

        with open(round_path) as f:
            rounds = json.load(f)
            if str(fl_round) in rounds.keys():
                return rounds[str(fl_round)]
            weight_path = f'{root_path}/current_weights.json'
            normalize_all_weights(weight_path)


            # Number of batches (rounds)

            #calc_num_batches = 3  # Assuming we want 4 rounds (25% of total sites per round)
            
            # Create balanced batches
            batches = create_balanced_batches(site_metrics, calc_num_batches)
            
            # Assign batches to rounds, ensuring Tick and Tack phases
            for round_num, batch in enumerate(batches,1):
                r1 = (fl_round-1)+(round_num*2)
                r2 = r1-1
                rounds[str(r2)] = batch
                rounds[str(r1)] = batch


            with open(round_path, 'w') as f:
                json.dump(rounds, f)
            with open(f'{round_path}_{fl_round}.json', 'w') as f:
                json.dump(rounds, f)

            return rounds[str(fl_round)]

def compute_label_dice_scores(site_metrics):
    labels = [1, 2, 4]  # Only the labels we have
    label_dice_scores = {label: {} for label in labels}
    for site, metrics in site_metrics.items():
        for label in labels:
            tensor_name = f'train_dice_per_label_{label}'
            if tensor_name in metrics:
                label_dice_scores[label][site] = metrics[tensor_name]
    
    return label_dice_scores

def rank_sites_by_label(label_dice_scores):
    labels = [1, 2, 4]  # Only the labels we have
    label_ranks = {label: [] for label in labels}
    for label, scores in label_dice_scores.items():
        sorted_sites = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label_ranks[label] = [site for site, score in sorted_sites]
    
    return label_ranks

def select_sites_for_batches(label_ranks, num_batches, sites_per_batch):
    batches = [[] for _ in range(num_batches)]
    used_sites=[]
    attempts=0

    for i in range(num_batches):        
        group = []          
        attempts=0     
        while len(batches[i]) < sites_per_batch and attempts < num_batches:
            for label in [1, 2, 4]:  # Only the labels we have
                usd = True
                while(usd):
                    if (len(label_ranks[label]) == 0):
                        break                  
                    req = label_ranks[label].pop(-1)
                    if req not in used_sites:
                        group.append(req)
                        used_sites.append(req)
                        usd = False
            attempts += 1
            #print(attempts, "/",num_batches)
            batches[i]=group
    # Reste werden wegeschmissen :/
    #for label in [1, 2, 4]:
    #    while(len(label_ranks[label])>0):
    #        req = label_ranks[label].pop(0)
    #        if req not in used_sites:
    #            used_sites.append(req)
    #            batches[0].append(req)
    return batches    
    

def create_balanced_batches(site_metrics, num_batches):
    sites_per_batch = max(1, len(site_metrics) // num_batches)
    #print(sites_per_batch)
    # Step 1: Compute average Dice scores for each label
    label_dice_scores = compute_label_dice_scores(site_metrics)
    #print(label_dice_scores)
    # Step 2: Rank sites by their Dice scores for each label
    label_ranks = rank_sites_by_label(label_dice_scores)
    #print(label_ranks)
    # Step 3: Select sites for each batch to ensure fair label representation
    batches = select_sites_for_batches(label_ranks, num_batches, sites_per_batch)
    #print(batches)
    return batches


## Example usage
#site_metrics = {
#    'site1': {'train_dice_per_label_0': 0.81, 'train_dice_per_label_1': 0.11, 'train_dice_per_label_2': 0.1, 'train_dice_per_label_4': 0.75},
#    'site2': {'train_dice_per_label_0': 0.82, 'train_dice_per_label_1': 0.12, 'train_dice_per_label_2': 0.18, 'train_dice_per_label_4': 0.8},
#    'site3': {'train_dice_per_label_0': 0.83, 'train_dice_per_label_1': 0.13, 'train_dice_per_label_2': 0.18, 'train_dice_per_label_4': 0.77},
#    'site4': {'train_dice_per_label_0': 0.51, 'train_dice_per_label_1': 0.81, 'train_dice_per_label_2': 0.81, 'train_dice_per_label_4': 0.8},
#    'site5': {'train_dice_per_label_0': 0.52, 'train_dice_per_label_1': 0.82, 'train_dice_per_label_2': 0.81, 'train_dice_per_label_4': 0.77},
#    'site6': {'train_dice_per_label_0': 0.53, 'train_dice_per_label_1': 0.83, 'train_dice_per_label_2': 0.81, 'train_dice_per_label_4': 0.77},
#    'site7': {'train_dice_per_label_0': 0.15, 'train_dice_per_label_1': 0.52, 'train_dice_per_label_2': 0.53, 'train_dice_per_label_4': 0.8},
#    'site8': {'train_dice_per_label_0': 0.16, 'train_dice_per_label_1': 0.53, 'train_dice_per_label_2': 0.52, 'train_dice_per_label_4': 0.77},
#    'site9': {'train_dice_per_label_0': 0.17, 'train_dice_per_label_1': 0.55, 'train_dice_per_label_2': 0.51, 'train_dice_per_label_4': 0.77},
#    'site110': {'train_dice_per_label_0': 0.17, 'train_dice_per_label_1': 0.55, 'train_dice_per_label_2': 0.51, 'train_dice_per_label_4': 0.77},
#    'site11': {'train_dice_per_label_0': 0.81, 'train_dice_per_label_1': 0.11, 'train_dice_per_label_2': 0.1, 'train_dice_per_label_4': 0.75},
#    'site12': {'train_dice_per_label_0': 0.82, 'train_dice_per_label_1': 0.12, 'train_dice_per_label_2': 0.18, 'train_dice_per_label_4': 0.8},
#    'site13': {'train_dice_per_label_0': 0.83, 'train_dice_per_label_1': 0.13, 'train_dice_per_label_2': 0.18, 'train_dice_per_label_4': 0.77},
#    'site14': {'train_dice_per_label_0': 0.51, 'train_dice_per_label_1': 0.81, 'train_dice_per_label_2': 0.81, 'train_dice_per_label_4': 0.8},
#    'site15': {'train_dice_per_label_0': 0.52, 'train_dice_per_label_1': 0.82, 'train_dice_per_label_2': 0.81, 'train_dice_per_label_4': 0.77},
#    'site16': {'train_dice_per_label_0': 0.53, 'train_dice_per_label_1': 0.83, 'train_dice_per_label_2': 0.81, 'train_dice_per_label_4': 0.77},
#    'site17': {'train_dice_per_label_0': 0.15, 'train_dice_per_label_1': 0.52, 'train_dice_per_label_2': 0.53, 'train_dice_per_label_4': 0.8},
#    'site18': {'train_dice_per_label_0': 0.16, 'train_dice_per_label_1': 0.53, 'train_dice_per_label_2': 0.52, 'train_dice_per_label_4': 0.77},
#    'site19': {'train_dice_per_label_0': 0.17, 'train_dice_per_label_1': 0.55, 'train_dice_per_label_2': 0.51, 'train_dice_per_label_4': 0.77},
#    'site110': {'train_dice_per_label_0': 0.17, 'train_dice_per_label_1': 0.55, 'train_dice_per_label_2': 0.51, 'train_dice_per_label_4': 0.77}
#
#}
#
##batches = create_balanced_batches(site_metrics, num_batches)
##print(batches)
#get_current_round(2, ['site1', 'site2', 'site3', 'site4', 'site5', 'site6', 'site7', 'site8', 'site9'], site_metrics)
