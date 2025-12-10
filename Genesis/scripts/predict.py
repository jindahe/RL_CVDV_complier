import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class AdvancedQuantumPredictor:
    def __init__(self, csv_path='experiment/four/result_logic_[abcd].csv'):
        # 存储结构: models[metric][nonzero_count] = [params]
        self.models = {} 
        self.global_models = {} # 备用模型（当某类数据不足时使用）
        self.metrics = ['single_gate', 'two_gate', 'total_gate', 'depth', 'latency']
        self.is_trained = False
        self.load_and_train(csv_path)

    def _parse_abcd(self, k_str):
        matches = re.findall(r'[+-]?\d+', str(k_str))
        if len(matches) != 4: return None
        return [int(m) for m in matches]

    def _get_features(self, abcd_list):
        """同时提取 阶数(Order) 和 非零个数(Count)"""
        if abcd_list is None: return None, None
        order = sum(abs(x) for x in abcd_list)
        nonzero_count = sum(1 for x in abcd_list if x != 0)
        return order, nonzero_count

    def _exp_model(self, x, A, B, C):
        """拟合模型 Y = A * e^(B*x) + C"""
        with np.errstate(over='ignore'):
            return A * np.exp(B * x) + C

    def load_and_train(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            # 1. 提取特征
            parsed_data = df['K=[abcd]'].apply(lambda x: self._get_features(self._parse_abcd(x)))
            df['order'] = parsed_data.apply(lambda x: x[0])
            df['nz_count'] = parsed_data.apply(lambda x: x[1])
            df = df.dropna(subset=['order', 'nz_count'])
            
            print("正在根据 [非零个数] 分组训练模型...")
            
            for metric in self.metrics:
                if metric not in df.columns: continue
                self.models[metric] = {}
                
                # --- A. 训练全局模型 (作为备用) ---
                try:
                    popt_global, _ = curve_fit(self._exp_model, df['order'], df[metric], 
                                             p0=[1, 0.2, 0], maxfev=5000)
                    self.global_models[metric] = popt_global
                except: pass

                # --- B. 分组训练 (核心逻辑) ---
                # 针对 1个非零项, 2个非零项, 3个, 4个 分别训练
                for nz in range(1, 5):
                    subset = df[df['nz_count'] == nz].sort_values('order')
                    
                    # 只有当数据点足够多(>=3)时才进行拟合
                    if len(subset) >= 3:
                        try:
                            # 针对每一组非零个数，拟合 order 与 metric 的关系
                            popt, _ = curve_fit(self._exp_model, subset['order'], subset[metric], 
                                              p0=[1, 0.2, 0], maxfev=5000)
                            self.models[metric][nz] = popt
                        except Exception as e:
                            print(f"  警告: {metric} 在非零项={nz} 时拟合失败")
            
            self.is_trained = True
            print("模型训练完成！已区分非零项个数 (1~4) 的差异。")
            
        except FileNotFoundError:
            print(f"错误: 找不到文件 {csv_path}")

    def predict(self, a, b, c, d):
        if not self.is_trained: return "模型未训练"

        # 1. 分析输入特征
        input_list = [a, b, c, d]
        order, nz_count = self._get_features(input_list)
        
        result = {
            'input': f'[{a}, {b}, {c}, {d}]',
            'features': f'Order={order}, NonZero={nz_count}'
        }
        
        # 2. 预测
        for metric in self.metrics:
            params = None
            used_model = "Specific"
            
            # 优先查找是否有针对该 非零个数(nz_count) 的特定模型
            if metric in self.models and nz_count in self.models[metric]:
                params = self.models[metric][nz_count]
            # 如果没有(例如数据不足)，回退到全局模型
            elif metric in self.global_models:
                params = self.global_models[metric]
                used_model = "Global(Fallback)"
            
            if params is not None:
                pred_val = self._exp_model(order, *params)
                result[metric] = round(pred_val, 2)
            else:
                result[metric] = None
                
        return result

    def plot_comparison(self, metric='single_gate'):
        """可视化：画出不同非零个数下的增长曲线"""
        if not self.is_trained: return
        
        plt.figure(figsize=(10, 6))
        x_range = np.linspace(0, 10, 100) # 假设阶数范围 0-10
        
        colors = ['blue', 'orange', 'green', 'red']
        found_data = False
        
        for nz in range(1, 5):
            if metric in self.models and nz in self.models[metric]:
                found_data = True
                params = self.models[metric][nz]
                y_pred = self._exp_model(x_range, *params)
                plt.plot(x_range, y_pred, label=f'Non-zero Count = {nz}', color=colors[nz-1], linewidth=2)
        
        if found_data:
            plt.title(f'Impact of Non-zero Elements on {metric}')
            plt.xlabel('Order (Sum of Abs)')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()
        else:
            print(f"没有足够的数据绘制 {metric} 的对比图")

# ================= 使用示例 =================

# 1. 初始化
predictor = AdvancedQuantumPredictor()

if predictor.is_trained:
    # 场景 A: 阶数相同 (Order=4)，但非零个数不同
    print("\n--- 对比：阶数相同，非零个数不同 ---")
    
    # Case 1: 只有一个非零项 (稀疏)
    # [4, 0, 0, 0] -> Order=4, NonZero=1
    res1 = predictor.predict(4, 0, 0, 0)
    print(f"输入 [4, 0, 0, 0]: {res1['single_gate']} (Gate)")
    
    # Case 2: 四个非零项 (密集)
    # [1, 1, 1, 1] -> Order=4, NonZero=4
    res2 = predictor.predict(1, 1, 1, 1)
    print(f"输入 [1, 1, 1, 1]: {res2['single_gate']} (Gate)")
    
    # Case 3: 两个非零项
    # [2, 2, 0, 0] -> Order=4, NonZero=2
    res3 = predictor.predict(2, 2, 0, 0)
    print(f"输入 [2, 2, 0, 0]: {res3['single_gate']} (Gate)")

    # 2. 画图查看差异
    predictor.plot_comparison('single_gate')