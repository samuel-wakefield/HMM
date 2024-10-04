from contextlib import contextmanager
from .data_loader import load_dice_data
from .printer import print_matrices
from .analyser import recall_score, precision_score, f1_score


@contextmanager
def should_work(msg=True):
    try:
        yield
    except Exception as e:
        if not hasattr(e, "should_work"):
            e.should_work = msg
        raise e


load_dice_data = should_work(False)(load_dice_data)
print_matrices = should_work(False)(print_matrices)
recall_score = should_work(False)(recall_score)
precision_score = should_work(False)(precision_score)
f1_score = should_work(False)(f1_score)
