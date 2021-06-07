def rgbToXyz(rgb):
    index = 0
    temp=list()
    for val in rgb:
        val = float(val) / 255
        if val > 0.04045:
            val = ((val + 0.055) / 1.055) ** 2.4
        else:
            val = val / 12.92
        temp.append(val * 100)
        index += 1
    temp = np.array(temp)
    xyzMat = np.array([[0.4124, 0.3576, 0.1805],[0.2126, 0.7152, 0.0722],[0.0193, 0.1192, 0.9505]])
    xyz = np.dot(xyzMat,temp)
    return xyz
def rgbToLab(rgb):
    xyz = rgbToXyz(rgb)
    xyz[0] = xyz[0]/95.047
    xyz[1] = xyz[1]/100.0
    xyz[2] = xyz[2]/108.883
    index = 0
    for val in xyz:
        if val > 0.008856452:
            val = val**0.333333
        else:
            val = (0.33333 * 23.36111 * val) + 0.137931
        xyz[index] = val
        index+=1
    L = 116 * xyz[1]-16
    a = 500 * (xyz[0]-xyz[1])
    b = 200 * (xyz[1]-xyz[2])
    return [L,a,b]
def rgb2lab(img):
    temp=list()
    for x in range(len(img)):
        temp.append(rgbToLab(img[x]))
    temp = np.array(temp)
    return temp