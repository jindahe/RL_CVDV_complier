import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class HighPrecisionQuantumPredictor:
    def __init__(self, csv_path='experiment/five/result_logic_[abcde].csv'):
        # 存储结构: models[metric][nz_count] = [params]
        self.models = {} 
        self.metrics = ['single_gate', 'two_gate', 'total_gate', 'depth', 'latency']
        self.is_trained = False
        self.load_and_train(csv_path)

    def _parse_abcde(self, k_str):
        matches = re.findall(r'[+-]?\d+', str(k_str))
        if len(matches) != 5: return None
        return [int(m) for m in matches]

    def _get_features(self, abcd_list):
        """
        提取三个特征：
        1. Order: 绝对值之和
        2. NZ_Count: 非零个数 (用于分组)
        3. Std: 非零绝对值的标准差 (用于区分 [2,2] 和 [3,1])
        """
        if abcd_list is None: return None, None, None
        
        # 只提取非零元素的绝对值
        non_zeros = [abs(x) for x in abcd_list if x != 0]
        
        order = sum(non_zeros)
        nz_count = len(non_zeros)
        
        # 计算标准差 (衡量数字之间的差异)
        # 如果只有一个数或没有数，标准差为 0
        std_val = np.std(non_zeros) if nz_count > 1 else 0.0
        
        return order, nz_count, std_val

    # --- 模型 1: 单变量模型 (用于 nz=1) ---
    def _model_simple(self, x, A, B, C):
        return A * np.exp(B * x) + C

    # --- 模型 2: 双变量模型 (用于 nz > 1) ---
    # X是一个包含两行的数组: X[0]=order, X[1]=std
    def _model_dual(self, X, A, B, C, D):
        order, std = X
        with np.errstate(over='ignore'):
            # 指数部分同时包含 Order 和 Std 的贡献
            return A * np.exp(B * order + D * std) + C

    def load_and_train(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            # 1. 提取所有特征
            features = df['K=[abcd]'].apply(lambda x: self._get_features(self._parse_abcd(x)))
            df['order'] = features.apply(lambda x: x[0])
            df['nz_count'] = features.apply(lambda x: x[1])
            df['std'] = features.apply(lambda x: x[2])
            
            df = df.dropna(subset=['order'])
            
            print("正在训练高精度模型 (引入 Abs 差异特征)...")
            
            for metric in self.metrics:
                if metric not in df.columns: continue
                self.models[metric] = {}
                
                # 按非零个数分组训练
                for nz in range(1, 6):
                    subset = df[df['nz_count'] == nz]
                    if len(subset) < 4: continue # 数据太少跳过
                    
                    try:
                        if nz == 1:
                            # 只有一个非零项，无需 Std 特征，使用简单模型
                            popt, _ = curve_fit(self._model_simple, subset['order'], subset[metric], 
                                              p0=[1, 0.2, 0], maxfev=5000)
                            self.models[metric][nz] = {'type': 'simple', 'params': popt}
                        else:
                            # 多个非零项，使用双变量模型 (Order + Std)
                            # 准备输入数据 (2, N)
                            X_data = np.vstack((subset['order'], subset['std']))
                            y_data = subset[metric]
                            
                            # p0: A=1, B=0.2(Order权重), C=0, D=0(Std权重，初始假设无影响)
                            popt, _ = curve_fit(self._model_dual, X_data, y_data, 
                                              p0=[1, 0.2, 0, 0], maxfev=10000)
                            self.models[metric][nz] = {'type': 'dual', 'params': popt}
                            
                    except Exception as e:
                        print(f"  警告: {metric} (NZ={nz}) 拟合失败: {e}")
            
            self.is_trained = True
            print("模型训练完成！")
            
        except FileNotFoundError:
            print(f"错误: 找不到文件 {csv_path}")

    def predict(self, a, b, c, d):
        if not self.is_trained: return {}

        # 1. 提取特征
        input_list = [a, b, c, d]
        order, nz_count, std_val = self._get_features(input_list)
        
        result = {
            'input': f'[{a}, {b}, {c}, {d}]',
            'details': f'Order={order}, NZ={nz_count}, Std={std_val:.2f}'
        }
        
        # 2. 预测
        for metric in self.metrics:
            if metric in self.models and nz_count in self.models[metric]:
                model_info = self.models[metric][nz_count]
                params = model_info['params']
                
                if model_info['type'] == 'simple':
                    pred_val = self._model_simple(order, *params)
                else:
                    # 双变量预测
                    pred_val = self._model_dual((order, std_val), *params)
                
                result[metric] = round(pred_val, 2)
            else:
                result[metric] = None
                
        return result

# ================= 验证脚本 =================

predictor = HighPrecisionQuantumPredictor()

if predictor.is_trained:
    # 场景 1: 均衡分布 [-2, 2, 0, 0]
    # 绝对值 [2, 2], Std Dev = 0
    res = predictor.predict(-5, 0 ,0, 0)
    print(f"   Single Gate: {res.get('single_gate')}")
    print(f"   Two Gate:    {res.get('two_gate')}")
    print(f"   Total Gate:  {res.get('total_gate')}")
    print(f"   Depth:       {res.get('depth')}")
    print(f"   Latency:     {res.get('latency')}")
    
   