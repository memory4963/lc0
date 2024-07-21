import os
import multiprocessing
import subprocess
import sys

def run(cmd):
    subprocess.call(cmd)

if __name__ == '__main__':
    cmd = sys.argv[2:]
    processes = []
    splits = int(sys.argv[1])
    print(cmd)
    for i in range(splits):
        process = multiprocessing.Process(target=run, args=(cmd + ['--parts', f'{i}/{splits}'],))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    print(cmd)
