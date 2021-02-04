import string

print(string.punctuation)
punctuations = string.punctuation
punctuations = punctuations + '’'
print(punctuations)


# function to remove
def remove_punc(text):
    text_punc_removed = [char for char in text if char not in punctuations]
    text_punc_removed_join = ''.join(text_punc_removed)

    return text_punc_removed_join
