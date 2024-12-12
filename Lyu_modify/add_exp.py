import torch
from transformers import AutoTokenizer, AutoModel

# 加载模型和分词器
model_name = "THUDM/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, output_hidden_states=True).eval()

# 示例输入
input_text = "405+20"
inputs = tokenizer(input_text, return_tensors="pt")

# 前向传播，获取每层隐藏状态
with torch.no_grad():
    outputs = model(**inputs)
hidden_states = outputs.hidden_states  # 每一层的隐藏状态

# 分析最后一位 Token 的隐藏状态逐层变化
final_token_idx = inputs['input_ids'].shape[1] - 1  # 最后一位索引
for layer_idx, hidden_state in enumerate(hidden_states):
    final_token_hidden = hidden_state[0, final_token_idx]  # 提取最后一位的隐藏状态
    logits = torch.matmul(final_token_hidden, model.lm_head.weight.T) + model.lm_head.bias  # 计算 logits
    probs = torch.softmax(logits, dim=-1)
    predicted_token_id = torch.argmax(probs).item()
    predicted_token = tokenizer.decode([predicted_token_id])
    print(f"Layer {layer_idx}: Predicted Token = {predicted_token}")
