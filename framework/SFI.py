from functools import cmp_to_key

# save pixel infomation
class PIXEL:
    def __init__(self, value, i, j):
        self.value = value # mask
        self.i = i # coordinate(i, j)
        self.j = j

# define compare function
def cmp(a, b):
    return b.value - a.value

# SFI identifies the significant pixels of the mfccs
def SFI(map):
    pixel_value = []
    for i in range(len(map)):
        for j in range(len(map[0])):
            pixel_value.append(PIXEL(map[i][j], i, j))
    # Sorting by value
    pixel_value = sorted(pixel_value, key = cmp_to_key(cmp)) 

    return pixel_value