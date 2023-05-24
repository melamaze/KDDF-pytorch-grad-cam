# save pixel infomation
class PIXEL:
    def __init__(self, value, i, j):
        self.value = value # mask
        self.i = i # coordinate(i, j)
        self.j = j

# FCM erase feature depends on gaussian filter 
def FCM(mfccs, gaussian_kernel, size_n, size_m):
    new_map = [[0.0 for i in range(size_m)] for j in range(size_n)]
    for i in range(size_n):
        for j in range(size_m):
            tmp = mfccs[i][j] * gaussian_kernel[1][1]
            if i - 1 >= 0 and j - 1 >= 0:
                tmp += mfccs[i - 1][j - 1] * gaussian_kernel[0][0]
            if i - 1 >= 0 and j + 1 < 100:
                tmp += mfccs[i - 1][j + 1] * gaussian_kernel[0][2]
            if i + 1 < 40 and j - 1 >= 0:
                tmp += mfccs[i + 1][j - 1] * gaussian_kernel[2][0]
            if i + 1 < 40 and j + 1 < 100:
                tmp += mfccs[i + 1][j + 1] * gaussian_kernel[2][2]
            if i - 1 >= 0:
                tmp += mfccs[i - 1][j] * gaussian_kernel[0][1]
            if i + 1 < 40:
                tmp += mfccs[i + 1][j] * gaussian_kernel[2][1]
            if j - 1 >= 0:
                tmp += mfccs[i][j - 1] * gaussian_kernel[1][0]
            if j + 1 < 100:
                tmp += mfccs[i][j + 1] * gaussian_kernel[1][2]
            new_map[i][j] = tmp 
    
    return new_map
