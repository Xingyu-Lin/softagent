import matplotlib.pyplot as plt
import csv

file1 = '/home/xingyu/Downloads/2.csv'
file2 = '/home/xingyu/Downloads/3.csv'


def readcsv(filename):
    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

            # get total number of rows
        print("Total no. of rows: %d" % (csvreader.line_num))
    data = list(map(list, zip(*rows)))
    return data


if __name__ == '__main__':
    # plt.figure()
    d1 = readcsv(file1)
    d2 = readcsv(file2)
    plt.figure()
    plt.plot(range(len(d1[2])), [float(x) for x in d1[2]], label='bc_update')
    plt.plot(range(len(d2[2])), [float(x) for x in d2[2]], label='curl')
    plt.legend()
    plt.show()
