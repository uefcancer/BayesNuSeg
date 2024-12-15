import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio("data/pannuke/", output="data/pannuke_dataset",
    seed=1337, ratio=(.80, .20), group_prefix=None, move=False) # default values
