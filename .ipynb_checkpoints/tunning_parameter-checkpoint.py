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
        if 'final best performance: ' in line:
            result = line
        if 'original testing performance:' in line:
            original = line
    return result, original


# confidence
# intervener_feature_number
# intervener_iteration
# intervener_reg
# intervener_learning_rate


dataset = 'Tools_and_Home_Improvement'

anchor_parameters = []
# --f_embedding_dim,  --batch_size, --reg, --learning_rate, --embedding_dim
# ' --f_embedding_dim ' + str(anchor_parameters[0]) + \
# ' --batch_size ' + str(anchor_parameters[1]) + \
# ' --reg ' + str(anchor_parameters[2]) + \
# ' --learning_rate ' + str(anchor_parameters[3]) + \
# ' --embedding_dim ' + str(anchor_parameters[4]) + \


print('------- confidence ------- ')
best_f1 = 0.0
for confidence in [-0.8, -0.9, -1.0, -1.1, -1.2, -1.3]:
    s = time.time()
    cmd = 'python3 main.py --anchor_model 2 ' \
          ' --data_path ./data/' + dataset + '/  ' \
          ' --intervener_feature_number 20 ' \
          ' --intervener_iteration 200 ' \
          ' --intervener_reg 0.01 ' \
          ' --intervener_learning_rate 0.1 ' \
          ' --confidence ' + str(confidence)
    print(cmd)
    sys.stdout.flush()
    result, original = sh(cmd)
    print(result)
    r = eval(result.split(':')[1])
    achieved = int(result.split(':')[2])
    print(original)
    print('confidence: ', confidence, r, achieved, 'time cost: ', str(time.time() - s))
    if r[2] > best_f1:
        best_f1 = r[2]
        best_confidence = confidence
    sys.stdout.flush()


print('------- learning_rate ------- ')
best_f1 = 0.0
for intervener_learning_rate in [0.1, 0.01, 0.2, 0.02, 0.05]:
    s = time.time()
    cmd = 'python3 main.py --anchor_model 2 ' \
          ' --data_path ./data/' + dataset + '/  ' \
          ' --intervener_feature_number 20 ' \
          ' --intervener_iteration 200 ' \
          ' --intervener_reg 0.01 ' \
          ' --intervener_learning_rate ' + str(intervener_learning_rate) + \
          ' --confidence ' + str(best_confidence)
    print(cmd)
    sys.stdout.flush()
    result, original = sh(cmd)
    r = eval(result.split(':')[1])
    achieved = int(result.split(':')[2])
    print(original)
    print('learning_rate: ', intervener_learning_rate, r, achieved, 'time cost: ', str(time.time() - s))
    if r[2] > best_f1:
        best_f1 = r[2]
        best_intervener_learning_rate = intervener_learning_rate
    sys.stdout.flush()


print('------- reg ------- ')
best_f1 = 0.0
for reg in [0.01, 0.1, 0.5, 1.0, 2.0]:
    s = time.time()
    cmd = 'python3 main.py --anchor_model 2 ' \
          ' --data_path ./data/' + dataset + '/  ' \
          ' --intervener_feature_number 20 ' \
          ' --intervener_iteration 200 ' \
          ' --intervener_reg ' + str(reg) + \
          ' --intervener_learning_rate ' + str(best_intervener_learning_rate) + \
          ' --confidence ' + str(best_confidence)
    print(cmd)
    sys.stdout.flush()
    result, original = sh(cmd)
    r = eval(result.split(':')[1])
    achieved = int(result.split(':')[2])
    print(original)
    print('reg: ', reg, r, achieved, 'time cost: ', str(time.time() - s))
    if r[2] > best_f1:
        best_f1 = r[2]
        best_reg = reg
    sys.stdout.flush()


print('------- feature_number ------- ')
best_f1 = 0.0
for feature_number in [1, 5, 10, 20, 50, 100]:
    s = time.time()
    cmd = 'python3 main.py --anchor_model 2 ' \
          ' --data_path ./data/' + dataset + '/  ' \
          ' --intervener_feature_number ' + str(feature_number) + \
          ' --intervener_iteration 200 ' \
          ' --intervener_reg ' + str(best_reg) + \
          ' --intervener_learning_rate ' + str(best_intervener_learning_rate) + \
          ' --confidence ' + str(best_confidence)
    print(cmd)
    sys.stdout.flush()
    result, original = sh(cmd)
    r = eval(result.split(':')[1])
    achieved = int(result.split(':')[2])
    print(original)
    print('feature_number: ', feature_number, r, achieved, 'time cost: ', str(time.time() - s))
    if r[2] > best_f1:
        best_f1 = r[2]
        best_feature_number = feature_number
    sys.stdout.flush()


print('------- iteration ------- ')
best_f1 = 0.0
for iteration in [50, 100, 200, 300, 500]:
    s = time.time()
    cmd = 'python3 main.py --anchor_model 2 ' \
          ' --data_path ./data/' + dataset + '/  ' \
          ' --intervener_feature_number ' + str(best_feature_number) + \
          ' --intervener_iteration '+ str(iteration) + \
          ' --intervener_reg ' + str(best_reg) + \
          ' --intervener_learning_rate ' + str(best_intervener_learning_rate) + \
          ' --confidence ' + str(best_confidence)
    print(cmd)
    sys.stdout.flush()
    result, original = sh(cmd)
    r = eval(result.split(':')[1])
    achieved = int(result.split(':')[2])
    print(original)
    print('iteration: ', iteration, r, achieved, 'time cost: ', str(time.time() - s))
    if r[2] > best_f1:
        best_f1 = r[2]
        best_iteration = iteration
    sys.stdout.flush()


print([best_feature_number, best_iteration, best_reg, best_intervener_learning_rate, best_confidence])



