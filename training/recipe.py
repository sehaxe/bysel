"""
⚙️ busel LOSS ENGINE v4.0 (MTP-4 STABILIZED)
Вычисляет многоголовый причинный лосс MTP-4 с затухающим взвешиванием.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from liger_kernel.transformers.functional import liger_cross_entropy
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False


def validate_training_schedule(max_steps, warmup_steps):
    """Runtime guard for the training schedule (ISSUES.md #7).

    Rejects `max_steps <= warmup_steps` (would cause NaN spikes in
    `autopilot.update_parameters` when computing `progress / 0` or `progress > 1.0`).
    Also rejects `warmup_steps < 1` (autopilot needs the first 50 steps to
    be free of predictive dampening).

    Returns the validated `(max_steps, warmup_steps)` as a tuple of ints, or
    raises `ValueError` with a helpful message.
    """
    if max_steps is None or warmup_steps is None:
        raise ValueError(
            f"max_steps ({max_steps}) and warmup_steps ({warmup_steps}) must be "
            f"set to integers before calling validate_training_schedule"
        )
    max_steps = int(max_steps)
    warmup_steps = int(warmup_steps)
    if max_steps <= warmup_steps:
        raise ValueError(
            f"max_steps ({max_steps}) must be strictly greater than "
            f"warmup_steps ({warmup_steps}). Either raise max_steps or "
            f"lower warmup_steps in configs/default.yaml."
        )
    if warmup_steps < 1:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be >= 1. The first 50 "
            f"steps also need to be free of predictive dampening (autopilot.py)."
        )
    return max_steps, warmup_steps


class buselLossEngine:
    def __init__(self, vocab_size=259):
        self.vocab_size = vocab_size

    def compute_pretrain_loss(self, logits, targets, mtp_logits_list=None, mtp_targets_list=None):
        """
        Вычисление основного лосса в низком разрешении (bfloat16 / float16).
        Интегрирован расчет потерь для дополнительных предсказательных голов MTP-4.
        """
        targets_device = targets.to(logits.device).long()
        
        # 1. Расчет основного лосса для головы t+1 (весовой коэффициент = 1.0)
        if HAS_LIGER and logits.device.type == "cuda":
            loss = liger_cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets_device.reshape(-1)
            )
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets_device.reshape(-1)
            )
            
        # 2. 🎯 РАСЧЕТ И ВЗВЕШИВАНИЕ ПОТЕРЬ MTP-4:
        # Если переданы логиты и таргеты для голов предсказания будущего (t+2, t+3, t+4),
        # мы рассчитываем лосс для каждой головы и суммируем их с затухающим коэффициентом.
        if mtp_logits_list is not None and mtp_targets_list is not None:
            # Веса важности предсказания каждого последующего шага (t+2, t+3, t+4)
            mtp_weights = [0.5, 0.25, 0.125]
            
            for i, (m_logits, m_targets) in enumerate(zip(mtp_logits_list, mtp_targets_list)):
                if m_logits is None or m_targets is None:
                    continue
                
                m_targets_device = m_targets.to(m_logits.device).long()
                
                if HAS_LIGER and m_logits.device.type == "cuda":
                    m_loss = liger_cross_entropy(
                        m_logits.reshape(-1, self.vocab_size),
                        m_targets_device.reshape(-1)
                    )
                else:
                    m_loss = F.cross_entropy(
                        m_logits.reshape(-1, self.vocab_size),
                        m_targets_device.reshape(-1)
                    )
                
                # Аккуратно добавляем взвешенную потерю головы к общему лоссу
                loss = loss + m_loss * mtp_weights[i]
                
        return loss

    def compute_sft_loss(self, logits, targets, thought_mask):
        masked_targets = targets.clone()
        masked_targets[thought_mask == 0] = -100
        
        mask = masked_targets != -100
        return F.cross_entropy(
            logits[mask].reshape(-1, self.vocab_size),
            masked_targets[mask].reshape(-1)
        )

    def compute_kto_loss(self, policy_logps, reference_logps, labels, beta=0.1, kl_weight=0.1):
        log_ratios = policy_logps - reference_logps
        kl = torch.clamp(log_ratios, min=0.0).mean()
        
        losses = []
        for log_ratio, label in zip(log_ratios, labels):
            if label == 1:
                losses.append(-F.logsigmoid(beta * (log_ratio - kl)))
            else:
                losses.append(-F.logsigmoid(beta * (kl - log_ratio)))
        
        kto_loss = torch.stack(losses).mean() + kl_weight * kl
        return kto_loss