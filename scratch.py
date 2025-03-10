import requests

url = "https://rog.asus.com/us/graphics-cards/graphics-cards/rog-astral/rog-astral-rtx5090-o32g-gaming/"  # 你要访问的网站
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print(response.text)  # 输出网页HTML内容
else:
    print(f"访问失败，状态码：{response.status_code}")
