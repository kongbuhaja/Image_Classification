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
    nohup python3 main.py --process val --path "$1" --cpus "0-19" --gpus "0" > "val.log" 2>&1 &

    sleep 10
}

wait_for_completion() {
    while pgrep -f "python3 main.py --process val --path $1 --cpus 0-19 --gpus 0" > /dev/null; do
        sleep 30
    done
}

task() {
    start_background_task "$1"
    wait_for_completion "$1"
}

log_dir="./logs"
log_directory_check "$log_dir"


dir="trained_models_dg8"

task "$dir/resnet18_1"
task "$dir/pasresnet18_1"

task "$dir/dresnet18_1"
task "$dir/dresnet18_2"
# task "$dir/dresnet18_3"
# task "$dir/dresnet18_4"
# task "$dir/dresnet18_5"
# task "$dir/dresnet18_6"
# task "$dir/dresnet18_7"
# task "$dir/dresnet18_8"
# task "$dir/dresnet18_9"

task "$dir/psdresnet18_1"
task "$dir/psdresnet18_2"
# task "$dir/psdresnet18_3"
# task "$dir/psdresnet18_4"
# task "$dir/psdresnet18_5"
# task "$dir/psdresnet18_6"
# task "$dir/psdresnet18_7"
# task "$dir/psdresnet18_8"
# task "$dir/psdresnet18_9"

task "$dir/psadresnet18_1"
task "$dir/psadresnet18_2"
# task "$dir/psadresnet18_3"
# task "$dir/psadresnet18_4"
# task "$dir/psadresnet18_5"
# task "$dir/psadresnet18_6"
# task "$dir/psadresnet18_7"
# task "$dir/psadresnet18_8"
# task "$dir/psadresnet18_9"

task "$dir/psddresnet18_1"
task "$dir/psddresnet18_2"
# task "$dir/psddresnet18_3"
# task "$dir/psddresnet18_4"
# task "$dir/psddresnet18_5"
# task "$dir/psddresnet18_6"
# task "$dir/psddresnet18_7"
# task "$dir/psddresnet18_8"
# task "$dir/psddresnet18_9"