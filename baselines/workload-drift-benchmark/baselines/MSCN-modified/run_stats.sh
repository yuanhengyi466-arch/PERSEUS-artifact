dataset=stats
for i in 1 2 3 4 5 6 7 8 9
do
CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_column_queries/${i}_column_queries/${i}_column_queries.csv --predict_query_file workloads/${dataset}/varying_column_queries/5_column_queries/5_column_queries.csv
CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_column_queries/${i}_column_queries/${i}_column_queries.csv --predict_query_file workloads/${dataset}/varying_column_queries/5_column_queries/5_column_queries.csv --robust
done

for i in 2 3 4 5 6
do
CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_table_queries/queries_${i}_tables/queries_${i}_tables.csv --predict_query_file workloads/${dataset}/varying_table_queries/queries_4_tables/queries_4_tables.csv
CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_table_queries/queries_${i}_tables/queries_${i}_tables.csv --predict_query_file workloads/${dataset}/varying_table_queries/queries_4_tables/queries_4_tables.csv --robust
done

for i in "7_2" "5_2" "3_2" "2_2" "1_1" "2_3" "2_5" "2_7"
do
CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_${i}/value_distribution_${i}.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2/value_distribution_7_2.csv 
CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_${i}/value_distribution_${i}.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2/value_distribution_7_2.csv --robust 
done

for i in "small" "large"
do
for j in "small" "large"
do
CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_label_queries/${i}_label_queries/${i}_label_queries.csv --predict_query_file workloads/${dataset}/varying_label_queries/${j}_label_queries/${j}_label_queries.csv 
CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_label_queries/${i}_label_queries/${i}_label_queries.csv --predict_query_file workloads/${dataset}/varying_label_queries/${j}_label_queries/${j}_label_queries.csv --robust 
done
done

#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv 
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_5_2.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv 
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_3_2.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv 
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_2_2.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv 
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_1_1.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv 
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_2_3.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv 
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_2_5.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv 
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_2_7.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv 


#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv --robust 
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_5_2.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv --robust
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_3_2.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv --robust
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_2_2.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv --robust
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_1_1.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv --robust
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_2_3.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv --robust
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_2_5.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv --robust
#CUDA_VISIBLE_DEVICES=3 python3 train.py --dataset ${dataset} --epochs 50 --cuda --train_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_2_7.csv --predict_query_file workloads/${dataset}/varying_value_distribution_queries/beta_distribution_2range/value_distribution_7_2.csv --robust 
