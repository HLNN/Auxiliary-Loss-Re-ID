import os
import sys
import time
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

i = 0
cmd_pre = 'cd /home/hlning/reid-strong-baseline;pwd;'
cmd = 'python tools/train.py --config_file=configs/softmax_triplet_with_center.yml MODEL.DEVICE_ID "(\'5\')" DATASETS.NAMES "(\'market1501\',\'msmt17\')" DATASETS.PROPS "(1., .3)"  DATASETS.ROOT_DIR "(\'/data/hln/datasets\')" OUTPUT_DIR "(\'logs/market1501-msmt17/30/\')"'
print(cmd)


def show_info(gpu_status):
    global i
    gpus = []
    for n, gpu in enumerate(gpu_status):
        gpus.append(f'Id: {n}\tMemory: {gpu["Memory"]} M\tPower: {gpu["Power"]} W')
    symbol = 'Waiting: ' + '>' * (i + 1) + ' ' * (10 - i - 1)
    i = (i + 1) % 8
    sys.stdout.write('\r' + ' | '.join(gpus + [symbol]))
    sys.stdout.flush()


def gpu_info():
    out_stream = os.popen('nvidia-smi -q -x').read()
    tree = ET.ElementTree(ET.fromstring(out_stream))
    root = tree.getroot()
    gpu_lists = [child for child in root if child.tag == 'gpu']
    gpu_status = []
    for n, gpu in enumerate(gpu_lists):
        children = {c.tag: c for c in gpu}
        fb_memory_usage_free = gpu[24][1].text.split(' ')[0]
        power_readings_power_draw = gpu[35][2].text.split(' ')[0]
        gpu_status.append({
            'Memory': float(fb_memory_usage_free),
            'Power': float(power_readings_power_draw),
        })
        pass
    show_info(gpu_status)
    return gpu_status


def narrow_setup(interval=1):
    f = True
    while f:  # set waiting condition
        gpu_status = gpu_info()
        for i, gpu in enumerate(gpu_status):
            if i == 0 and gpu['Memory'] < 10000 and gpu['Power'] < 10000:
                f = False
                break
            time.sleep(interval)
    print('\n', i, cmd)
    os.system(cmd_pre + cmd.format(id=i))


if __name__ == '__main__':
    narrow_setup()
    print(cmd)
