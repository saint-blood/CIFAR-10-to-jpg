import imageio
import numpy as np
import pickle

# 对数据集进行解压缩
def unpickle(file):
    shujuji = open(file, 'rb')
    dict = pickle.load(shujuji, encoding='latin1')
    shujuji.close()
    return dict

# 生成训练集图片，如果需要其他格式，只需要改图片后缀名即可。
for j in range(1, 6):
    dataName = "data_batch_" + str(j)  # 此处一定要把数据集的12345放在与本py一起的一个文件夹内。
    xuliehua = unpickle(dataName)#文件序列化
    print(dataName + " is loading...")

    for i in range(0, 10000):
        img = np.reshape(xuliehua['data'][i], (3, 32, 32))  #xuliehua['data']为图片二进制数据
        img = img.transpose(1, 2, 0)  # 读取image
        picName = 'train/' + str(xuliehua['labels'][i]) + '_' + str(i + (j - 1)*10000) + '.jpg'  # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
        imageio.imwrite(picName, img)
    print(dataName + " loaded.")

print("test_batch is loading...")

# 生成测试集图片
testxlh = unpickle("test_batch")
for i in range(0, 10000):
    img = np.reshape(testxlh['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'test/' + str(testxlh['labels'][i]) + '_' + str(i) + '.jpg'
    imageio.imwrite(picName, img)
print("test_batch loaded.")
