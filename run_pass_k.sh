ROOT=/home/user/EvoCodeBench

Source_Code_Root=$ROOT/Source_Code
Dependency_Root=$ROOT/Dependency_Data

tasks=(baseline)
models=(codellama-7b_greedy deepseek-7b_greedy)

for task in ${tasks[@]}; do
    for model in ${models[@]}; do
        echo "Running pass@1 for $task $model"
        python check_source_code.py Source_Code
        python pass_k.py \
            --output_file $ROOT/model_completion/$task/$model/completion.jsonl \
            --log_file $ROOT/model_completion/$task/$model/test_results.jsonl \
            --source_code_root $Source_Code_Root \
            --k 1 \
            --n 1
    done
done