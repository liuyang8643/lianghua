"""The prior validation-based refiner is retired and preserved in its study archive."""

def run(root):
    raise RuntimeError("Legacy validation-based refinement is disabled by the training-only policy")

def admit_final(root, frozen):
    if frozen.get("selection_split") != "train":
        raise ValueError("Only training-selected configurations may enter observation")
