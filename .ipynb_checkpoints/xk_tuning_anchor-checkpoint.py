import multiprocessing as mp
import numpy as np
import copy
import os
import time


class GpuSchedular:
    def __init__(self, n_gpus=2):
        self.n_gpus = 2
        self.reset()
        
    def allocate(self, task_idx):
        """
        负责给GPU分配任务，尽量使得GPU任务数量尽量平均。不考虑显存大小。只是平均分任务。
        """
        assert(task_idx not in self.task2gpu)
        n_tasks = [len(val) for key, val in self.gpu2tasks.items()]
        idx = (np.array(n_tasks)*-1).argmax()  # 选择最小的
        target = [i for i in self.gpu2tasks.keys()][idx] # 选择对应的gpu id
        self.gpu2tasks[target].append(task_idx)
        self.task2gpu[task_idx] = target
        return target
    
    def reset(self):
        self.gpu2tasks = {i: [] for i in range(self.n_gpus)}
        self.task2gpu = {}
    

class TaskPool:
    def __init__(self, n_pool=5, arg_type='long', n_gpu=2):
        self.n_pool = n_pool
        self.arg_type = arg_type
        self.gpu_schedular = GpuSchedular(n_gpu)
        
        self.reset()
    
    def reset(self):
        self.best_param_dict = {}
    
    def collect_results(self, info):
        """
        key, val ? 
        """
        return 0, 12
    
    def _worker(self, param_dict, gpu_id):
        cmd = "CUDA_VISIBLE_DEVICES={}".format(gpu_id) + " " + self._cmd_body() + " " + self._paramdict2str(param_dict)
        print (cmd + "\n")
        r = os.popen(cmd)
        info = r.readlines()
        return self.collect_results(info)
        
    def start(self, grid_param_dict):
        """
        grad_param_dict: 
            'anchor_model: [4]'
            'reg': [0.001, 0.01, 0.1]
        """
        s = time.time()
        for key, vals in grid_param_dict.items():
            results = [None for _ in range(len(vals))]
            with mp.Pool(self.n_pool) as p:
                for idx, val in enumerate(vals):
                    tmp_dict = copy.deepcopy(self.best_param_dict)
                    tmp_dict.update({key:val})
                    results[idx] = p.apply_async(self._worker, (tmp_dict, self.gpu_schedular.allocate(idx)))
                
                output_keys = [None for _ in range(len(vals))]
                output_vals = [None for _ in range(len(vals))]
                for idx, res in enumerate(results): 
                    output_keys[idx], output_vals[idx] = res.get()
                self.gpu_schedular.reset()
                
                print (output_keys)
                idx = np.array(output_keys).argmax()
                print(idx)
                print ("Find Better {}={}: {}".format(key, vals[idx], output_vals[idx]))
                self.best_param_dict[key] = vals[idx]
                
        print ('Time: \t', (time.time() - s), ' sec')
        print ("Best Parameters: \n{}".format(self.best_param_dict))

    def _paramdict2str(self, dic):
        params = []
        slash = "--" if self.arg_type == "long" else "-"
        for key, val in dic.items():
            assert (type(key) == str)
            params.append(slash + key)
            params.append(str(val))
        return " ".join(params)
    
    def _cmd_body(self):
        """
        将param_dict中的参数展开作为 --key val 的形式
        """
        return "head main.py"
        
    
class AnchorTaskPool(TaskPool):
    def __init__(self, n_pool, arg_type):
        super(AnchorTaskPool, self).__init__(n_pool, arg_type)
        
    def _cmd_body(self):
        #return "python main.py"
        return "python main.py --data_path ./data/Digital_Music/ --anchor_model 1 --intervener_batch_size 736 "
    
    def collect_results(self, info):
        parameter, result, result_reward, time_cost = '', '', '', ''
        generate_info = ""
        for line in info:
            if 'anchor best performance: ' in line:
                result = line
            if 'final generated sample number' in line:
                generate_info = line
        if result != "": 
            r = eval(result.split(':')[1])
        else:
            r = (0,0,0,0)
        return r[2], (result, generate_info)
    
    
    
# 单元测试
gpu_schedular = GpuSchedular(2)
assert (gpu_schedular.allocate('task_1') == 0)
assert (gpu_schedular.allocate('task_2') == 1)
assert (gpu_schedular.allocate('task_3') == 0)
assert (gpu_schedular.allocate('task_4') == 1)
assert (gpu_schedular.allocate('task_5') == 0)
assert (gpu_schedular.allocate('task_6') == 1)
print  (gpu_schedular.gpu2tasks)
print  (gpu_schedular.task2gpu)


anchor_grid_param_dict = {
    'reg': [0.0001, 0.0005, 0.0025, 0.0125, 0.0625, 0.3, 1.5], 
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.5], 
    'embedding_dim': [10, 30, 50, 80, 100], 
    'f_embedding_dim': [10, 30, 50, 80, 100], 
}

# Tuning for anchor model 
task = AnchorTaskPool(12, 'long')
task.start(anchor_grid_param_dict)
#task.start(anchor_grid_param_dict) # for better initial values