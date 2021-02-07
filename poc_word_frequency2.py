from collections import Counter
test = 'Lenovo teases what might be the first true Lenovo'
freq_distribution = Counter(test.split()).most_common()
print(freq_distribution)