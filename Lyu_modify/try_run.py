from transformers import AutoTokenizer, AutoModel  

# 加载本地模型和分词器
local_model_dir = "/home/ubuntu/.cache/huggingface/hub/models--THUDM--chatglm3-6b/snapshots/e9e0406d062cdb887444fe5bd546833920abd4ac" 
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)  
model = AutoModel.from_pretrained(local_model_dir, trust_remote_code=True, device='cuda').eval()
response, history = model.chat(tokenizer, "你好", history=[])  
print(response)
response, history = model.chat(tokenizer, "介绍一下chatglm的模型结构", history=[])
print(response)
