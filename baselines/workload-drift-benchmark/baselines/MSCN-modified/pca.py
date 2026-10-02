from mscn.data import load_data
from mscn.util import *
from dataset import *

train_query_file, min_max_file, predict_query_file, result_file, save_folder, \
    train_bitmap_file, predict_bitmap_file, tables = load_job_dataset()
train_joins, train_predicates, train_tables, train_samples, train_labels = load_data()