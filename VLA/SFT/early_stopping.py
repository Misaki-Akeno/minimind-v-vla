import numpy as np
from collections import deque
from typing import Optional


class EarlyStopping:
    """
    早停工具类，基于准确率的滑动平均
    """
    
    def __init__(self, window_size: int = 20, threshold: float = 0.95, patience: int = 5):
        """
        Args:
            window_size: 滑动窗口大小
            threshold: 准确率阈值
            patience: 连续满足阈值的次数才触发早停
        """
        self.window_size = window_size
        self.threshold = threshold
        self.patience = patience
        
        self.accuracies = deque(maxlen=window_size)
        self.consecutive_count = 0
        self.best_accuracy = 0.0
        self.should_stop = False
        
    def update(self, accuracy: float) -> bool:
        """
        更新准确率并检查是否应该早停
        
        Args:
            accuracy: 当前验证准确率
            
        Returns:
            是否应该早停
        """
        self.accuracies.append(accuracy)
        self.best_accuracy = max(self.best_accuracy, accuracy)
        
        # 只有当窗口填满时才开始检查
        if len(self.accuracies) < self.window_size:
            return False
            
        # 计算滑动平均
        moving_avg = np.mean(self.accuracies)
        
        print(f"验证准确率: {accuracy:.4f}, {self.window_size}步滑动平均: {moving_avg:.4f} (阈值: {self.threshold})")
        
        # 检查是否超过阈值
        if moving_avg >= self.threshold:
            self.consecutive_count += 1
            print(f"连续满足阈值次数: {self.consecutive_count}/{self.patience}")
            
            if self.consecutive_count >= self.patience:
                self.should_stop = True
                print(f"触发早停！滑动平均准确率 {moving_avg:.4f} >= {self.threshold} 已连续 {self.patience} 次")
                return True
        else:
            self.consecutive_count = 0
            
        return False
    
    def get_moving_average(self) -> Optional[float]:
        """获取当前滑动平均值"""
        if len(self.accuracies) == 0:
            return None
        return float(np.mean(self.accuracies))
    
    def reset(self):
        """重置早停状态"""
        self.accuracies.clear()
        self.consecutive_count = 0
        self.best_accuracy = 0.0
        self.should_stop = False
