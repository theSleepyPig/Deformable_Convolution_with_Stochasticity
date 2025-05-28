import os
import time
import subprocess

# 要监控的PID列表
pids_to_monitor = [48215, 48132]

# 检查进程是否还在运行
def is_running(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

# 监控进程
def monitor_processes(pids):
    while any(is_running(pid) for pid in pids):
        time.sleep(120)

# 运行指定的命令
def run_command():
    command = "nohup python train.py > train_output.log 2>&1 &"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    monitor_processes(pids_to_monitor)
    run_command()