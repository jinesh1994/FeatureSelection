# for char in 'jenil'.lower():
#     print(ord(char) - 96)


def alphabet_position(text):
    nums = [str(ord(x) - 96) for x in text.lower() if x >= 'a' and x <= 'z']
    return " ".join(nums)


print(alphabet_position('jenil'))
