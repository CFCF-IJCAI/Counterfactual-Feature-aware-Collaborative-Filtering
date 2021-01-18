import subprocess
import numpy as np
import sys
import time

def run(command):
    subprocess.call(command, shell=True)

def sh(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    parameter, result, result_reward, time_cost = '', '', '', ''
    for line in iter(p.stdout.readline, b''):
        line = line.rstrip().decode('utf8')
        if 'final best performance:' in line:
            result = line
    return result


datasets = ['Automotive', 'Amazon_Instant_Video', 'Baby', 'Beauty', 'Cell_Phones_and_Accessories',
            'Digital_Music', 'Grocery_and_Gourmet_Food', 'Home_and_Kitchen', 'Musical_Instruments', 'Office_Products',
            'Patio_Lawn_and_Garden', 'Pet_Supplies', 'Tools_and_Home_Improvement', 'Toys_and_Games', 'Video_Games', 'yelp']


print('------- batch_size ------- ')
for data in datasets:
    cmd = 'python3 ./model/IF_Qua.py ' \
          ' --data_path ./data/' + data + '/'
    print(cmd)
    sys.stdout.flush()
    result = sh(cmd)
    print('result: ', result)
    sys.stdout.flush()

    cmd = 'python3 ./model/UF_Att.py ' \
          ' --data_path ./data/' + data + '/'
    print(cmd)
    sys.stdout.flush()
    result = sh(cmd)
    print('result: ', result)
    sys.stdout.flush()

