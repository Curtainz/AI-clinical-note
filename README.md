已实现：将./input.json的内容读取，传入基于bionic的模型，获取./output.json

Bionic本地部署模型：qwen3-4b-instruct-2507

配置：Automatic Optimize Based on Hardware  
CPU Thread Pool Size: 8  
         Temperature: 0.7  
               Top K: 20  
               Top P: 0.8  
               Min P: 0  
                Seed: 42  

基础URL：http://localhost:1235/v1

当./input.json存在时，快速运行：
```powershell
Copy-Item -LiteralPath '.\prompts\baseline.txt' -Destination '.\prompts\active.txt'
& .\.venv\Scripts\python.exe .\main.py
```

尚待实现：训练、输出校验。

