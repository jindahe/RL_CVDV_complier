import numpy as np
import sympy as sp
from typing import List

def compute_integer_nullspace(A: np.ndarray) -> np.ndarray:
    """
    计算矩阵 A (ker(A)) 的整数零空间基。
    
    逻辑步骤:
    1. 使用 SymPy 计算有理数零空间。
    2. 计算所有分母的最小公倍数 (LCM)，将向量转换为整数。
    3.除以最大公约数 (GCD) 使向量“原始化” (Primitive)。
    4. 调整符号，使得第一个非零元素为正。
    5. (特定逻辑) 如果基向量超过1个，添加首尾向量的组合以形成闭环（用于QAOA混合层）。
    
    Args:
        A (np.ndarray): 形状为 (m, d) 的整数约束矩阵。
        
    Returns:
        np.ndarray: 整数零空间基向量组成的矩阵。
    """
    # 1. 转换为 SymPy 矩阵并计算零空间
    A_sym = sp.Matrix(A)
    ns_rational = A_sym.nullspace()
    
    if not ns_rational:
        raise ValueError("Nullspace empty; constraints overconstrained.")
    
    # 2. 收集所有分母并计算 LCM
    denoms = []
    for vec in ns_rational:
        for entry in vec:
            if hasattr(entry, 'is_Rational') and entry.is_Rational:
                denoms.append(entry.q)  # 获取分母
    
    lcm_den = sp.lcm(denoms) if denoms else sp.Integer(1)
    
    # 3. 转换为整数向量并进行归一化处理
    int_vecs = []
    for vec in ns_rational:
        # 乘上 LCM 转为整数
        int_vec = (lcm_den * vec).applyfunc(lambda x: int(x))
        # 展平为 numpy 数组
        int_vec = np.array(int_vec).flatten().astype(int)
        
        # 除以 GCD (使向量互素)
        gcd = np.gcd.reduce(int_vec)
        if gcd != 0:
            int_vec = int_vec // gcd
            
        # 4. 符号归一化：确保第一个非零元素为正
        nonzero_indices = np.nonzero(int_vec)[0]
        if len(nonzero_indices) > 0 and int_vec[int(nonzero_indices[0])] < 0:
            int_vec = -int_vec
            
        int_vecs.append(int_vec)
    
    # 去重
    unique_vecs = set(tuple(v) for v in int_vecs)
    # 转换回 numpy 数组，注意：set 是无序的，但通常我们需要确定的顺序，
    # 这里为了保持和原代码一致直接转 list 再转 array
    unique_vecs = np.array(list(unique_vecs))
    
    # 5. 添加额外的“环路”向量 (Circle Loop Logic)
    # 原代码逻辑：取第一个和最后一个向量组合，增加连通性
    if len(unique_vecs) > 1:
        first_vec = unique_vecs[0]
        last_vec = unique_vecs[-1]
        
        vec_sum = first_vec + last_vec
        vec_diff = first_vec - last_vec
        
        # 选择非零元素较少的组合（更稀疏的向量通常更好）
        if len(np.nonzero(vec_sum)[0]) < len(np.nonzero(vec_diff)[0]):
            new_vec = vec_sum
        else:
            new_vec = vec_diff
            
        # 对新向量进行归一化 (GCD 和 符号)
        gcd = np.gcd.reduce(new_vec)
        if gcd != 0:
            new_vec = new_vec // gcd
            
        nonzero_indices = np.nonzero(new_vec)[0]
        if len(nonzero_indices) > 0 and new_vec[int(nonzero_indices[0])] < 0:
            new_vec = -new_vec
            
        # 将新向量堆叠到基底中
        unique_vecs = np.vstack([unique_vecs, new_vec])
    
    return unique_vecs

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
   
    A_example = np.array([[1, 1, -1, -1]])
    
    print(f"约束矩阵 A:\n{A_example}\n")
    
    try:
        basis = compute_integer_nullspace(A_example)
        print(f"计算出的整数零空间基 (包含环路向量):\n{basis}")
        
    except Exception as e:
        print(f"Error: {e}")