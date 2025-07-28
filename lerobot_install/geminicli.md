
# 设置代理环境变量（替换为您的实际代理地址和端口）
$env:HTTP_PROXY = "http://127.0.0.1:10804" \
$env:HTTPS_PROXY = "http://127.0.0.1:10804" \
$env:http_proxy = "http://127.0.0.1:10804" \
$env:https_proxy = "http://127.0.0.1:10804"

# 运行Gemini CLI
gemini
