import numpy as np
def log_probability(I_t, I_hat, sigma):
    return np.log(1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((I_t - I_hat) ** 2) / (2 * sigma ** 2)))

def total_log_probability(I, t1, t2, sigma):
    total_log_prob = (t2 - t1) * np.log(sigma) + np.sum(log_probability(I[t1:t2], np.mean(I[t1:t2]), sigma))
    return total_log_prob

def compare_log_probabilities(t1, t2, t3, I, sigma):
    logp_t1_t2 = total_log_probability(I, t1, t2, sigma[t1])
    logp_t2_t3 = total_log_probability(I, t2, t3, sigma[t2])
    logp_t3_t1 = total_log_probability(I, t3, t1, sigma[t3])
    return logp_t1_t2 + logp_t2_t3 - logp_t3_t1

def find_level_transitions(I, sigma, threshold):
    transitions = []
    t1 = 0
    t3 = len(I) - 1
    while t3 - t1 > 1:
        min_logp = float('inf')
        min_t2 = None
        for t2 in range(t1 + 1, t3):
            logp = compare_log_probabilities(t1, t2, t3, I, sigma)
            if logp < min_logp:
                min_logp = logp
                min_t2 = t2
        if min_logp < threshold:
            transitions.append(min_t2)
            t1 = min_t2
        else:
            t3 += 1
    return transitions

def calculate_sigma(I, window_size):
    sigma = []
    for i in range(len(I)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(I), i + window_size // 2)
        sigma.append(np.std(I[start_index:end_index]))
    return sigma


# 使用示例:
data_list=[] # 电信号数据
window_size = 100  # 计算sigma的窗口
sigma = calculate_sigma(data_list[:1000], window_size)   # 取前一千个点看看结果
print("s",sigma)
threshold = -50
transitions = find_level_transitions(data_list[:1000], sigma, threshold)
print("Level transitions found at:", transitions)
