import matplotlib.pyplot as plt

# 数据
scores = {"PRISM": 1,
          "TITAN": 0,
          "ChatPath": 0}

# 提取名字和分数
names = list(scores.keys())
values = list(scores.values())

# 绘制柱状图，设置柱子宽度
plt.figure(figsize=(6, 4))
plt.bar(names, values, color='skyblue', width=0.3)  # width 调整柱宽

# 添加标题和标签
plt.title("BLEU_test_image")
plt.ylabel("score")
plt.xticks(rotation=45, ha="right")  # 如果名字长，旋转一下

# 显示数值
for i, v in enumerate(values):
    plt.text(i, v + 0.02, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig("fig.jpg")
plt.show()
