ROOT=/home/user/EvoCodeBench

Source_Code_Root=$ROOT/Source_Code
Dependency_Root=$ROOT/Dependency_Data

tasks=(baseline)
models=(codellama-7b_greedy deepseek-7b_greedy)

for task in ${tasks[@]}; do
    for model in ${models[@]}; do
        echo "Running recall@1 for $task $model"
        python ../check_source_code.py $ROOT/Source_Code
        python recall_k.py \
            --output_file $ROOT/model_completion/$task/$model/completion.jsonl \
            --log_file $ROOT/model_completion/$task/$model/dependency_results.jsonl \
            --k 1 \
            --source_code_root $Source_Code_Root \
            --dependency_data_root $Dependency_Root \
            --data_file $ROOT/data.jsonl
    done
done