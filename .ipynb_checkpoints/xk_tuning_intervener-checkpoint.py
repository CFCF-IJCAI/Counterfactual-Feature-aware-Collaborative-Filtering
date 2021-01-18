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

        
import pynvml
pynvml.nvmlInit()
class AvgMemoryGpuSchedular:
    def __init__(self, n_gpus=2):
        self.n_gpus = 2
        self.reset()
        
    def get_used_memory(self, index):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return meminfo.used // 1024 // 1024

    def allocate(self, task_idx):
        """
        负责给GPU分配任务，尽量使得GPU任务数量尽量平均。
        计算显存大小，然后给显存最小的进行传递任务。需要额外安装库： pip install nvidia-ml-py3
        """
        import time
        time.sleep(20)
        assert(task_idx not in self.task2gpu)
        used_memorys = [self.get_used_memory(gpu_id) for gpu_id in range(self.n_gpus)]
        #print (used_memorys)
        idx = (np.array(used_memorys)*-1).argmax()  # 选择最小的
        target = idx # gpu_idx is the target gpu
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
        self.gpu_schedular = AvgMemoryGpuSchedular(n_gpu)
        
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
                
                for ret_idx, (ret_key, ret_val) in enumerate(zip(output_keys, output_vals)):
                    print ("[Summary]{}:{}:\n\t{}".format(key, vals[ret_idx], ret_val))
                    
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
        return "python main.py --data_path ./data/Amazon_Instant_Video/ --anchor_model 1 --intervener_batch_size 736 "
    
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


class IntervenerBalancedTaskPool(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Digital_Music/ --anchor_model 4 --intervener_batch_size 18000 "
    def default_start(self):
        self.best_param_dict = {'confidence': 0.693, 'intervener_learning_rate': 0.0001, 'intervener_reg': 0.0001, 'reg': 0.0025, 'learning_rate': 0.001}
        default_param_dict = {
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9], 
            'reg': [0.0001, 0.0005, 0.0025, 0.0125, 0.0625, 0.3, 1.5], 
            'intervener_learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9], 
            'intervener_reg': [0.0001, 0.0005, 0.0025, 0.0125, 0.0625, 0.3, 1.5],
            'confidence': [0.693, 0.65, 0.60, 0.55, 0.50, 0.45, 0.4],
            'intervener_feature_number': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
        }
        self.start(default_param_dict)


class Intervener3TaskPool(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Office_Products/ --anchor_model 3 --intervener_batch_size 12000 --intervener_soft True "
    def default_start(self):
        self.best_param_dict = {'confidence': 0.693, 'intervener_learning_rate': 0.0001, 'intervener_reg': 0.0001, 'reg': 0.0025, 'learning_rate': 0.001}
        default_param_dict = {
            'intervener_learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9], 
            'intervener_reg': [0.0001, 0.0005, 0.0025, 0.0125, 0.0625, 0.3, 1.5],
        }
        self.start(default_param_dict)

        
        
class Intervener4TaskPool(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Amazon_Instant_Video/ --anchor_model 4 --intervener_batch_size 600 --intervener_soft True "
    def default_start(self):
        self.best_param_dict = {'confidence': 0.55, 'intervener_learning_rate': 0.5, 'intervener_reg': 1.5, 'reg': 0.3, 'learning_rate': 0.0001, 'intervener_feature_number': 60, 'intervener_l1_reg': 0.0025}
        default_param_dict = {
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9], 
            'reg': [0.0001, 0.0005, 0.0025, 0.0125, 0.0625, 0.3, 1.5], 
            'intervener_learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9], 
            'intervener_reg': [0.0001, 0.0005, 0.0025, 0.0125, 0.0625, 0.3, 1.5],
            'intervener_l1_reg': [0.0001, 0.0005, 0.0025, 0.0125, 0.0625, 0.3, 1.5],
            'confidence': [0.693, 0.65, 0.60, 0.55, 0.50, 0.45, 0.4],
        }
        self.start(default_param_dict)
        #self.start(default_param_dict)


class IntervenerTaskPool(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Tools_and_Home_Improvement/ --anchor_model 2 --intervener_batch_size 2000 "
    def default_start(self):
        self.best_param_dict = {'confidence': 0.55, 'intervener_learning_rate': 0.5, 'intervener_reg': 1.5, 'reg': 0.3, 'learning_rate': 0.0001, 'intervener_feature_number': 60}

        default_param_dict = {
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.5], 
            'reg': [0.0001, 0.0005, 0.0025, 0.0125, 0.0625, 0.3, 1.5], 
            'intervener_learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9], 
            'intervener_reg': [0.0001, 0.0005, 0.0025, 0.0125, 0.0625, 0.3, 1.5],
            'confidence': [0.693, 0.65, 0.60, 0.55, 0.50, 0.45, 0.4],
            'intervener_feature_number': [20, 40, 60, 80, 100], 
        }
        self.start(default_param_dict)
    
    
class SoftOnlyTaskPool(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Amazon_Instant_Video/ --anchor_model 2 --intervener_batch_size 700 --intervener_soft True "
    def default_start(self):
        self.best_param_dict = {'confidence': 0.55, 'intervener_learning_rate': 0.5, 'intervener_reg': 1.5, 'reg': 0.3, 'learning_rate': 0.0001, 'intervener_l1_reg': 0.0001, 'intervener_feature_number': 60}
        
        default_param_dict = {
            'confidence': [0.693],
        }
        self.start(default_param_dict)
    
    
class WithoutPerturbationTaskPool(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Amazon_Instant_Video/ --anchor_model 4 --intervener_batch_size 600 --intervener_soft True --intervener_l1_reg 0.0" 
    def default_start(self):
        self.best_param_dict = {'confidence': 0.55, 'intervener_learning_rate': 0.5, 'intervener_reg': 1.5, 'reg': 0.3, 'learning_rate': 0.0001, 'intervener_l1_reg': 0.0001, 'intervener_feature_number': 60}
        
        
        self.best_param_dict['intervener_l1_reg'] = 0.0
        default_param_dict = {
            'intervener_learning_rate': [0.5],
        }
        self.start(default_param_dict)
    
    
class WithoutNoiseControlTaskPool(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Amazon_Instant_Video/ --anchor_model 2 --intervener_batch_size 700 --confidence 1000.0" 
    def default_start(self):
        self.best_param_dict =  {'confidence': 0.693, 'intervener_learning_rate': 0.5, 'intervener_reg': 0.0001, 'reg': 0.0001, 'learning_rate': 0.01, 'intervener_l1_reg': 0.0025}
        default_param_dict = {
            'confidence': [1000.0],
        }
        self.start(default_param_dict)
        

class ParameterAnalysis10TaskPool(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Amazon_Instant_Video/ --anchor_model 4 --intervener_batch_size 600 "

    def default_start(self):
        self.best_param_dict = {'confidence': 0.55, 'intervener_learning_rate': 0.5, 'intervener_reg': 1.5, 'reg': 0.3, 'learning_rate': 0.0001, 'intervener_l1_reg': 0.0001, 'intervener_feature_number': 60}
 
        default_param_dict = {
            'confidence': [0.693, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3],
        }
        self.start(default_param_dict)
    
    
class ParameterAnalysis11TaskPool(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Amazon_Instant_Video/ --anchor_model 4 --intervener_batch_size 600 "

    def default_start(self):
        self.best_param_dict = {'confidence': 0.55, 'intervener_learning_rate': 0.5, 'intervener_reg': 1.5, 'reg': 0.3, 'learning_rate': 0.0001, 'intervener_l1_reg': 0.0001, 'intervener_feature_number': 60}

        default_param_dict = {
            'intervener_feature_number': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
        }
        self.start(default_param_dict)
        
        
class ParameterAnalysis12TaskPool(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Amazon_Instant_Video/ --anchor_model 4 --intervener_batch_size 600 --intervener_soft True "
        #return "python main.py --data_path ./data/Digital_Music/ --anchor_model 4 --intervener_batch_size 18000 --intervener_soft True "

    def default_start(self):
        #assert False, "Input the best_param_dict"
        #self.best_param_dict = {'confidence': 0.65, 'intervener_learning_rate': 0.3, 'intervener_reg': 0.0005, 'reg': 0.0025, 'learning_rate': 0.01, 'intervener_l1_reg': 0.0025}
        
        self.best_param_dict = {'confidence': 0.55, 'intervener_learning_rate': 0.001, 'intervener_reg': 0.0125, 'reg': 0.0005, 'learning_rate': 0.0001, 'intervener_feature_number': 60, 'intervener_l1_reg': 0.0025}

        default_param_dict = {
            'intervener_l1_reg': [0.0001, 0.0005, 0.0025, 0.0125, 0.0625, 0.3, 0.5, 0.7, 0.9, 1.0],
        }
        self.start(default_param_dict)


class CaseFindTask(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Tools_and_Home_Improvement/ --anchor_model 2 --intervener_batch_size 2300 --intervener_iteration 1000 "

    def default_start(self):
        # assert False, "Input the best_param_dict"
        self.best_param_dict = {'confidence': 0.55,'intervener_learning_rate': 0.5, 'intervener_reg': 1.5, 'reg': 0.3, 'learning_rate': 0.0001, 'intervener_l1_reg': 0.0001, 'intervener_feature_number': 60}
        
        default_param_dict = {
            "case_model": range(0, 280), 
        }
        self.start(default_param_dict)

        
class BalancedTunningHard(AnchorTaskPool):
    def _cmd_body(self):
        return "python main.py --data_path ./data/Tools_and_Home_Improvement/ --anchor_model 4 --intervener_batch_size 2400"
    def default_start(self):
        self.best_param_dict = {'confidence': 0.693, 'intervener_learning_rate': 0.5, 'intervener_reg': 0.0001, 'reg': 0.0001, 'learning_rate': 0.01, 'intervener_l1_reg': 0.0025}
        
        default_param_dict = {
            #'balanced_multiply': [0.5, 0.7, 0.9, 1, 1.1, 1.3, 1.5, 2.0],
            'balanced_multiply': [1, 1.3, 1.5, 1.7, 1.9, 2.0],
        }
        self.start(default_param_dict)


        
id2task = {
    0: CaseFindTask(20, "long"), # case find. need range from [0, #Feature]
    
    6: IntervenerTaskPool(10, 'long'), # hard
    3: Intervener4TaskPool(3, 'long'),  # soft + balanced
    2: Intervener3TaskPool(5, 'long'),  # random noise
    1: IntervenerBalancedTaskPool(10, 'long'), # hard + balanced
    
    
    7: SoftOnlyTaskPool(10, 'long'),  # soft - balanced
    8: WithoutPerturbationTaskPool(10, 'long'), # with K = |F|
    9: WithoutNoiseControlTaskPool(10, 'long'), # with noise control: confidence = inf
    
    
    10:ParameterAnalysis10TaskPool(10, 'long'), # confident changes
    11:ParameterAnalysis11TaskPool(10, 'long'), # K changes
    12:ParameterAnalysis12TaskPool(10, 'long'), # beta changes
    
    13:BalancedTunningHard(10, 'long'), # Balanced Fine Tune
}
    
import sys
assert (len(sys.argv) == 2)
print ("tune model: ", sys.argv[1])
tune_model = int(sys.argv[1])
id2task[tune_model].default_start()