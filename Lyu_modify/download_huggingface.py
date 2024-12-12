from huggingface_hub import snapshot_download

# 模型名称和本地路径
model_name = "THUDM/chatglm3-6b"
local_dir = "./models/chatglm3-6b"

# 下载模型文件
snapshot_download(repo_id=model_name, cache_dir=local_dir)