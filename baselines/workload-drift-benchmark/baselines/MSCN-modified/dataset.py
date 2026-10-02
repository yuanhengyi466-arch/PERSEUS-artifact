def load_job_dataset():
    save_folder = "./workloads/job-light-ranges"
    query_folder = save_folder + "/value_distribution_join/beta_distribution_2range"
    #query_folder = save_folder
    train_query_file = query_folder + "/value_distribution_1_1.csv"
    min_max_file = save_folder + "/" 
    predict_query_file = query_folder + "/value_distribution_7_2.csv"
    result_file = save_folder + "/result.csv"
    train_bitmap_file = query_folder + "/value_distribution_1_1_bitmap.csv"
    predicate_bitmap_file = query_folder + "/value_distribution_7_2_bitmap.csv"
    tables = ['cast_info', 'movie_companies', 'movie_info_idx', 'movie_info', 'movie_keyword', 'title']
    return train_query_file, min_max_file, predict_query_file, result_file, save_folder, \
           train_bitmap_file, predicate_bitmap_file, tables

def load_stats_dataset():
    save_folder = "./workloads/stats"
    query_folder = save_folder + "/varying_value_distribution_queries/beta_distribution_2range"
    #train_query_file = query_folder + "/value_distribution_7_2.csv"
    train_query_file = save_folder + "/target_prediction.csv"
    #train_query_file = save_folder + "/adaptation_100.csv"
    min_max_file = save_folder + "/" 
    #predict_query_file = query_folder + "/value_distribution_7_2.csv"
    predict_query_file = save_folder + "/target_prediction.csv"
    result_file = save_folder + "/result.csv"
    #adaptation_file = save_folder + "/cbp_500.csv"
    #adaptation_file = save_folder + "/cbp_10.csv"
    adaptation_file = query_folder + "/adaptation_100.csv"
    validation_file = query_folder + "/validation.csv"
    #train_bitmap_file = query_folder + "/value_distribution_7_2_bitmap.csv"
    train_bitmap_file = save_folder + "/target_prediction_bitmap.csv"
    #train_bitmap_file = save_folder + "/adaptation_100_bitmap.csv"
    #predicate_bitmap_file = query_folder + "/value_distribution_7_2_bitmap.csv"
    predicate_bitmap_file = save_folder + "/target_prediction_bitmap.csv"
    adaptation_bitmap_file = query_folder + "/adaptation_100_bitmap.csv"
    #adaptation_bitmap_file = save_folder + "/adaptation_all_bitmap.csv"
    #adaptation_bitmap_file = save_folder + "/cbp_10_bitmap.csv"
    #tables = ['cast_info', 'movie_companies', 'movie_info_idx', 'movie_info', 'movie_keyword', 'title']
    tables = ['cast_info', 'movie_companies', 'movie_info_idx', 'movie_info', 'movie_keyword', 'title']
    return train_query_file, min_max_file, predict_query_file, result_file, save_folder, \
            train_bitmap_file, predicate_bitmap_file, tables


def load_higss_dataset():
    save_folder  = "./workloads/higss"
    train_query_file = "./workloads/higss/train_3.csv"
    min_max_file = save_folder + "/"
    predict_query_file = "./workloads/higss/adaptation_3_prediction.csv"
    result_file = "./workloads/higss/result.csv"
    save_folder = "./workloads/higss"
    adaptation_file = "./workloads/higss/adaptation_3_adaptation.csv"
    validation_file = "./workloads/higss/validation.csv"
    train_bitmap_file = "./workloads/higss/train_3_bitmap.csv"
    predicate_bitmap_file = "./workloads/higss/adaptation_3_prediction_bitmap.csv"
    adaptation_bitmap_file = "./workloads/higss/adaptation_3_adaptation_bitmap.csv"
    tables = ['higss']
    return train_query_file, min_max_file, predict_query_file, result_file, save_folder, \
            adaptation_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables

def load_higgs_full_dataset():
    save_folder = "./workloads/higgs_full"
    train_query_file = save_folder + "/train_queries_5000_2.csv"
    #train_query_file = save_folder + "/adaptation_5.csv"
    min_max_file = "./workloads/higgs_full/"
    predict_query_file = save_folder + "/prediction_5000.csv"
    result_file = save_folder + "/result.csv"
    adaptation_file = save_folder + "/adaptation_2.csv"
    validation_file = save_folder + "/validation.csv"
    train_bitmap_file = save_folder + "/train_queries_5000_2_bitmap.csv"
    #train_bitmap_file = save_folder + "/adaptation_2_bitmap.csv"
    predicate_bitmap_file = save_folder + "/prediction_5000_bitmap.csv"
    adaptation_bitmap_file = save_folder + "/adaptation_2_bitmap.csv"
    tables = ['higgs']
    return train_query_file, min_max_file, predict_query_file, result_file, save_folder, \
            adaptation_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables

def load_forest_dataset():
    save_folder = "./workloads/forest"
    #train_query_file = save_folder + "/mix_adaptation_queries_5000.csv"
    #train_query_file = save_folder + "/mix_adaptation_queries_10.csv"
    train_query_file = save_folder + "/adaptation_queries_5000.csv"
    min_max_file = "./workloads/forest/"
    predict_query_file = save_folder + "/mix_prediction_queries_5000.csv"
    result_file = save_folder + "/result.csv"
    adaptation_file = save_folder + "/mix_adaptation_queries_10.csv"
    validation_file = save_folder + "/mix_adaptation_queries_10.csv"
    #train_bitmap_file = save_folder + "/mix_adaptation_queries_5000_bitmap.csv"
    #train_bitmap_file = save_folder + "/mix_adaptation_queries_10_bitmap.csv"
    train_bitmap_file = save_folder + "/adaptation_queries_5000_bitmap.csv"
    predicate_bitmap_file = save_folder + "/mix_prediction_queries_5000_bitmap.csv"
    adaptation_bitmap_file = save_folder + "/mix_adaptation_queries_10_bitmap.csv"
    tables = ['forest']
    return train_query_file, min_max_file, predict_query_file, result_file, save_folder, \
            adaptation_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables
