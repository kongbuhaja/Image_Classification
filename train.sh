log_directory_check(){
    log_dir=$1
    if [ -d "$log_dir" ]; then
        :
    else
        mkdir -p "$log_dir"
    fi
}

start_background_task() {
    nohup python3 main.py --model "$1" > "$2/$1.log" 2>&1 &
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
task resnet18 "$log_dir"
task dresnet18 "$log_dir"

task psaresnet18 "$log_dir"
task psdresnet18 "$log_dir"
task psadresnet18 "$log_dir"
task psddresnet18 "$log_dir"

task c2psaresnet18 "$log_dir"
task c2psdresnet18 "$log_dir"
task c2psadresnet18 "$log_dir"
task c2psddresnet18 "$log_dir"
