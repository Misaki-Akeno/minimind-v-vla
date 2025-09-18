import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(model, val_loader, device, max_eval_steps=None):
    """
    在验证集上评估模型
    
    Args:
        model: 待评估的模型
        val_loader: 验证数据加载器
        device: 设备
        max_eval_steps: 最大评估步数，None表示评估整个验证集
        
    Returns:
        dict: 包含各种指标的字典
    """
    model.eval()
    total_samples = 0
    correct_actions = 0
    total_lm_loss = 0.0
    total_action_loss = 0.0
    num_batches = 0
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader, desc="验证中")):
            if max_eval_steps and step >= max_eval_steps:
                break
                
            X = batch['input_ids'].to(device)
            Y = batch['labels'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            action = batch['action'].to(device)
            
            # 前向传播
            res = model(X, pixel_values=pixel_values)
            
            # 语言建模损失
            loss_lm_tok = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss_lm = (loss_lm_tok * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            
            # 动作预测
            last_hidden = res.last_hidden_state[:, -1, :]  # [B, H]
            ah = model.module.action_head if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.action_head
            action_logits = ah(last_hidden)  # [B, 9]
            loss_action = F.cross_entropy(action_logits, action)
            
            # 计算动作准确率
            pred_actions = action_logits.argmax(dim=-1)
            correct_actions += (pred_actions == action).sum().item()
            total_samples += action.size(0)
            
            # 累积损失
            total_lm_loss += loss_lm.item()
            total_action_loss += loss_action.item()
            num_batches += 1
    
    # 计算平均指标
    avg_lm_loss = total_lm_loss / num_batches if num_batches > 0 else 0.0
    avg_action_loss = total_action_loss / num_batches if num_batches > 0 else 0.0
    action_accuracy = correct_actions / total_samples if total_samples > 0 else 0.0
    
    metrics = {
        'action_accuracy': action_accuracy,
        'avg_lm_loss': avg_lm_loss,
        'avg_action_loss': avg_action_loss,
        'total_samples': total_samples,
        'correct_actions': correct_actions
    }
    
    return metrics
