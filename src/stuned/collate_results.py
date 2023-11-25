import sys
import numpy as np
import wandb 

from tqdm import tqdm

sys.path.append(".")

KEYWORDS = ["accuracy", "loss"]
LINKS_FILE = "wandb_links.txt"

def main():

    run_links = []
    with open(LINKS_FILE, "r") as f:
        for line in f:
            run_links.append(line.strip())

    api = wandb.Api()

    all_results = []
    
    tokens = run_links[0].split('/')
    entity = tokens[3]
    project = tokens[4]
    wandb_id = tokens[-1]
    run = api.run(f"{entity}/{project}/{wandb_id}")
    keys = [key for key in run.summary.keys() if any([keyword in key for keyword in KEYWORDS])]

    for link in tqdm(run_links):
        wandb_id = link.split('/')[-1]
        run = api.run(f'{entity}/{project}/{wandb_id}')
        result = run.summary
        for key in keys:
            if 'accuracy' in key:
                result[key] = 100 * result[key]
        all_results.append(result)
    
    # calculate mean and std for each key
    collated_results = {}
    for key in keys:
        collated_results[key] = {
            "mean": np.mean([result[key] for result in all_results]),
            "std": np.std([result[key] for result in all_results])
        }

    # print in a pretty way: key: mean ± std
    print(f"Collated results for {len(run_links)} runs:")
    for key in keys:
        print(f"{key}: {collated_results[key]['mean']:.3f} ± {collated_results[key]['std']:.3f}")

if __name__ == "__main__":
    main()