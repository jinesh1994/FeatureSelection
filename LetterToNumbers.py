# for char in 'jenil'.lower():
#     print(ord(char) - 96)


def letter_to_number(text):
    nums = [str(ord(x) - 96) for x in text.lower() if x >= 'a' and x <= 'z']
    return " ".join(nums)


def number_to_letter(classified_data):
    labels = []
    for data in classified_data:
        labels.append(chr(int(data) + 96))
    return labels


# print(letter_to_number('jenil'))
# print(number_to_letter([10, 5, 14, 9, 12]))
