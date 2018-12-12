import pandas as pd


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


# Returns data row wise - 1st column has labels
def pickDataClass(filename, class_ids):
    load_file = pd.read_csv(filename, sep=",", header=None)
    load_file = load_file.transpose()
    result = []
    for i in class_ids:
        for j in load_file.values:
            if int(j[0]) == int(i):
                result.append(j)
    result = pd.DataFrame(result)
    return result

# print(letter_to_number('jenil'))
# print(number_to_letter([10, 5, 14, 9, 12]))
