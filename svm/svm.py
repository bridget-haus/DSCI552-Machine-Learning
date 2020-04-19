import matplotlib
import matplotlib.pyplot as plt


def read_file(file):
    '''Read the classification.txt file. Output a list of lists.'''
    fd = open(file, "r")
    rows = fd.read().splitlines()
    data_list = []
    for row in rows:
        row2 = row.replace("+", "")
        row_list = row2.split(",")
        row_list_nums = []
        for item in row_list:
            val = float(item)
            if val == 1.0:
                val = int(val)
            elif val == -1.0:
                val = 0
            row_list_nums.append(val)
        data_list.append(row_list_nums)
    return data_list


def plot_data(data_list):
    '''Read the cleaned data, and plot the labels in 2D scatterplot'''
    feat1 = []
    feat2 = []
    labels = []
    colors = ['red','blue']
    for row in data_list:
        feat1.append(row[0])
        feat2.append(row[1])
        labels.append(row[2])
    fig = plt.figure(figsize=(8,8))
    plt.scatter(feat1, feat2, c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()


def main():
    file1 = 'linsep.txt'
    file2 = 'nonlinsep.txt'
    input1 = read_file(file1)
    input2 = read_file(file2)
    plot_data(input1)
    plot_data(input2)

if __name__ == "__main__":
    main()