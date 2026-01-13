from datasets import load_dataset

ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")
print(ds)
print(ds["train"][0])
