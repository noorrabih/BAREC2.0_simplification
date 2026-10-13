# from barec 10M get 20 samples from each level and save to csv
import pandas as pd
import random

# CAMeL-Lab/BAREC-Corpus-v1.0

# open this data first
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("CAMeL-Lab/BAREC-Corpus-v1.0")

source_readability_counts = (
    ds['dev'].to_pandas()
    .pivot_table(index='Source', columns='Readability_Level_19', aggfunc='size', fill_value=0)
)

source_readability_counts.to_csv("source_readability_counts.csv", header=True)

import numpy as np

level_targets = {
    8: 6, 9: 7, 10: 6, 11: 6, 12: 13, 13: 12, 14: 13,
    15: 12, 16: 6, 17: 7, 18: 6, 19: 6,
}
assert sum(level_targets.values()) == 100

bands = [[8, 9, 10, 11], [12, 13], [14, 15], [16, 17, 18, 19]]

df = ds['dev'].to_pandas()

# one row per sentence, keeping all columns
sentences = df.drop_duplicates(subset='ID')

rng = np.random.RandomState(42)


def allocate_level(level_df, n, used_sources, rng):
    """Pick n sentences from level_df, round-robin across sources,
    preferring sources not yet used earlier in this band."""
    sources_at_level = level_df['Source'].unique().tolist()
    new_sources = [s for s in sources_at_level if s not in used_sources]
    old_sources = [s for s in sources_at_level if s in used_sources]
    rng.shuffle(new_sources)
    rng.shuffle(old_sources)
    priority = new_sources + old_sources

    pools = {
        src: level_df[level_df['Source'] == src].sample(frac=1, random_state=rng.randint(1_000_000))
        for src in priority
    }
    pointers = {src: 0 for src in priority}

    chosen_idx = []
    while len(chosen_idx) < n:
        progressed = False
        for src in priority:
            if len(chosen_idx) >= n:
                break
            pool = pools[src]
            ptr = pointers[src]
            if ptr < len(pool):
                chosen_idx.append(pool.index[ptr])
                pointers[src] += 1
                progressed = True
        if not progressed:
            break  # not enough sentences left at this level across all sources

    chosen = level_df.loc[chosen_idx]
    return chosen, set(chosen['Source'].unique())


sampled_ids = []

for band in bands:
    used_sources = set()
    for level in band:
        n = level_targets[level]
        level_df = sentences[sentences['Readability_Level_19'] == level]
        chosen, used_this_level = allocate_level(level_df, n, used_sources, rng)
        sampled_ids.extend(chosen['ID'].tolist())
        used_sources |= used_this_level

sample_sentences = sentences[sentences['ID'].isin(sampled_ids)]
print(len(sample_sentences))


sample_sentences.groupby(['Readability_Level_19', 'Source']).size().to_csv("sample_sentences_counts.csv", header=True)

sample_sentences.to_csv("/Users/nour.rabih/Desktop/sample_sentences.csv", index=False, header=True)