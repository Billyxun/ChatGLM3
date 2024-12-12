import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

# 加载本地模型和分词器
local_model_dir = "/home/ubuntu/.cache/huggingface/hub/models--THUDM--chatglm3-6b/snapshots/e9e0406d062cdb887444fe5bd546833920abd4ac"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(local_model_dir, trust_remote_code=True, output_hidden_states=True).to(device).eval()

# 示例输入样例
input_examples = [
    "1+1",       # 简单加法，无进位
    "2+3",       # 简单加法，无进位
    "405+20",    # 跨位进位
    "999+1",     # 全位进位
    "12345+6789" # 多位复杂加法
]

# 存储每个样例的结果
results = {}

# 对每个输入样例进行分析
for input_text in input_examples:
    # 分词和前向传播
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 确保输入张量在设备上
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states  # 获取每层的隐藏状态

    # 存储当前输入的每层预测结果
    layer_results = []

    # 分析每一层的输出
    for layer_idx, hidden_state in enumerate(hidden_states):
        # 修正 final_token_idx
        final_token_idx = min(inputs['input_ids'].shape[1] - 1, hidden_state.shape[1] - 1)
        final_token_hidden = hidden_state[0, final_token_idx].to(device)  # 移动到同一设备
        # logits = torch.matmul(final_token_hidden, model.lm_head.weight.T.to(device)) + model.lm_head.bias.to(device)  # 计算 logits
        # 替代 lm_head 的处理方式
        logits = torch.matmul(final_token_hidden, model.transformer.word_embeddings.weight.T.to(device))
        probs = torch.softmax(logits, dim=-1)
        predicted_token_id = torch.argmax(probs).item()
        predicted_token = tokenizer.decode([predicted_token_id])
        layer_results.append(predicted_token)  # 存储该层的预测结果

    # 将结果存入字典
    results[input_text] = layer_results

# 打印每个样例的结果
for input_text, layer_results in results.items():
    print(f"Input: {input_text}")
    for layer_idx, result in enumerate(layer_results):
        print(f"  Layer {layer_idx}: {result}")

# 绘图：每层的末尾数字变化过程
plt.figure(figsize=(12, 6))
for input_text, layer_results in results.items():
    # 提取末尾数字的变化（只取数字部分）
    end_digits = [int(r[-1]) if r[-1].isdigit() else -1 for r in layer_results]
    plt.plot(range(len(layer_results)), end_digits, label=f"Input: {input_text}")

plt.xlabel("Layer Index")
plt.ylabel("Last Digit")
plt.title("Last Digit Evolution Across Layers")
plt.legend()
plt.grid(True)
plt.show()
