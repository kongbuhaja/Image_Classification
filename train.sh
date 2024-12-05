log_directory_check(){
    log_dir=$1
    if [ -d "$log_dir" ]; then
        :
    else
        mkdir -p "$log_dir"
    fi
}

start_background_task() {
    echo "$1"
    nohup python3 main.py --model "$1" > "$2/$1.log" 2>&1 &
    sleep 10
}

wait_for_completion() {
    while pgrep -f "python3 main.py --model $1" > /dev/null; do
        sleep 30
    done
}

task() {
    start_background_task "$1" "$2"
    wait_for_completion "$1"
}

log_dir="./logs"
log_directory_check "$log_dir"

# task resnet18 "$log_dir"
# task dresnet18 "$log_dir"
# task psdresnet18 "$log_dir"
# task psadresnet18 "$log_dir"
# task psddresnet18 "$log_dir"

# task dresnet18 "$log_dir"
task dresnet182 "$log_dir"
# task dresnet18 "$log_dir"
task dresnet182 "$log_dir"

# task psdresnet18 "$log_dir"
task psdresnet182 "$log_dir"
# task psdresnet18 "$log_dir"
task psdresnet182 "$log_dir"

# task dresnet182 "$log_dir"
# task psdresnet182 "$log_dir"
# task psadresnet182 "$log_dir"
# task psddresnet182 "$log_dir"

# task c2psaresnet18 "$log_dir"
# task c2psdresnet18 "$log_dir"
# task c2psadresnet18 "$log_dir"
# task c2psddresnet18 "$log_dir"

# task resnet18 "$log_dir"
# task dresnet18 "$log_dir"

# # # task psaresnet18 "$log_dir"
# task psdresnet18 "$log_dir"
# task psadresnet18 "$log_dir"
# task psddresnet18 "$log_dir"

# task c2psaresnet18 "$log_dir"
# task c2psdresnet18 "$log_dir"
# task c2psadresnet18 "$log_dir"
# task c2psddresnet18 "$log_dir"

# log_directory_check(){
#     log_dir=$1
#     if [ -d "$log_dir" ]; then
#         :
#     else
#         mkdir -p "$log_dir"
#     fi
# }

# start_background_task() {
#     nohup python3 main.py --path "$1" --process val > "1.log" 2>&1 &
# }

# wait_for_completion() {
#     while pgrep -f "python3 main.py --path $1 --process val" > /dev/null; do
#         sleep 10
#     done
# }

# task() {
#     start_background_task "$1" "$2"
#     wait_for_completion "$1"
# }

# log_dir="./logs"
# log_directory_check "$log_dir"
# # task resnet18 "$log_dir"
# # task dresnet18 "$log_dir"
# # task resnet182 "$log_dir"
# # task dresnet182 "$log_dir"

# # task psaresnet18 "$log_dir"
# # task psdresnet18 "$log_dir"
# # task psadresnet18 "$log_dir"
# # task psddresnet18 "$log_dir"

# # task c2psaresnet18 "$log_dir"
# # task c2psdresnet18 "$log_dir"
# # task c2psadresnet18 "$log_dir"
# # task c2psddresnet18 "$log_dir"


# # task trained_models/resnet18_1 "$log_dir"
# # task trained_models/dresnet18_1 "$log_dir"
# # task trained_models/resnet182_1 "$log_dir"
# # task trained_models/dresnet182_1 "$log_dir"

# # task trained_models/psaresnet18_1 "$log_dir"
# # task trained_models/psdresnet18_1 "$log_dir"
# # task trained_models/psadresnet18_1 "$log_dir"
# # task trained_models/psddresnet18_1 "$log_dir"

# # task trained_models/c2psaresnet18_1 "$log_dir"
# task trained_models/c2psdresnet18_1 "$log_dir"
# task trained_models/c2psadresnet18_1 "$log_dir"
# task trained_models/c2psddresnet18_1 "$log_dir"