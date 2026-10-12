from token_counter import count_tokens


text = "学校成立于1950年，是一所综合性大学。"

count = count_tokens(text)

print("文本：", text)
print("Token 数量：", count)

