import cv2
import hashlib
import numpy as np
import random
from skimage.metrics import structural_similarity as ssim


# Wong 脆弱水印算法

"""
函数说明:
euclid , extended_euclid, mod_inverse, generate_key_pair 都是RSA需要的一些函数
RSA_encrypt,RSA_decrypt 是RSA的加密和解密函数,用于计算10进制数字
binary_to_int , int_to_binary, 用于RSA计算前后将二进制与10进制数字之间进行转换
全局变量 p,q 是精心挑选的, 最好不要修改,这是本实验设计的一个无法避免的会出现误差的地方
全局变量 public_key , private_key为公私钥, 每次运行此程序, 一般公私钥是不同的
这是因为 RSA选定公私钥的时候就并不是唯一的

prikey_encrypt,      用私钥对一维二进制流签名
pubkey_decrypt,      用公钥对一维二进制流解密
H,                   哈希函数
devide_blocks,       划分图像为小块
mark_block,          对小块进行加水印操作
put_watermark,       对图像加水印,即先划小块,将小块施加水印再合并起来
get_block_mark,      对每一个小块提取水印
extract_watermark,   对图像提取水印,即先划小块,将小块提取水印再合并起来
generate_chessboard, 生成棋盘状的Br图像,即背景图
generate_white,      生成全白色的Br图像,即背景图

以下仅为测试用的函数:
alter_SQ_pixels,     修改含水印图像左上角区域的像素
alter_pixels,        修改含水印图像某一随机部分的像素
paste_image,         将一张图像粘贴到另一张图像的某个位置
ctrl_c,              区域复制粘贴
gaussian,            高斯噪声攻击
salt_pepper,         椒盐噪声攻击
psnr_cal,            计算两张图像的PSNR值
ssim_cal,            计算两张图像的SSIM值
"""


# 欧几里德算法
def euclid(a, b):
    while b != 0:
        a, b = b, a % b
    return a


# 扩展的欧几里德算法
def extended_euclid(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x, y = extended_euclid(b % a, a)
    return gcd, y - (b // a) * x, x


# 求模逆
def mod_inverse(a, m):
    euclid, x, _ = extended_euclid(a, m)
    if euclid != 1:
        raise ValueError("Modular inverse does not exist.")
    return x % m


# 生成 RSA 公私钥对
def generate_key_pair(p, q):
    n = p * q
    phi = (p - 1) * (q - 1)
    e = random.randint(2, phi - 1)
    while euclid(e, phi) != 1:
        e = random.randint(2, phi - 1)
    d = mod_inverse(e, phi)
    return (e, n), (d, n)


# RSA 加密
def RSA_encrypt(en_key, plaintext):
    e, n = en_key
    ciphertext = [pow(int(ch), e, n) for ch in plaintext]
    return ciphertext


# RSA 解密
def RSA_decrypt(de_key, ciphertext):
    d, n = de_key
    plaintext = [pow(int(ch), d, n) for ch in ciphertext]
    return plaintext


"""
选择两个素数,19*3449 = 65531,而16位二进制能达到的最大数字为 65535
由于RSA自身的数学性质,还是会有 5/65535 *100% = 0.0076% 的部分无法被正确解密出来
这 0.0076% 看起来很小,但很偶尔的情况下是会在提取的水印中显现出来的,就是细小的黑灰线
虽然这两个数的安全性几乎像纸糊的一样,但我们也不打算在此处使用更安全的 1024 位或更大的素数
因为 python 进行数据计算太慢了, 我们这里只是简单地演示一下算法的原理就好
这里考虑到计算速度,考虑到 16 位一分组,以及尽量减小误差,所以选择了这两个数
"""
p = 19
q = 3449

public_key, private_key = generate_key_pair(p, q)


def binary_to_int(binary_array):
    int_array = []
    for i in range(0, len(binary_array), 16):
        num = 0
        for digit in binary_array[i:i+16]:
            num = (num << 1) | digit
        int_array.append(num)
    return int_array


def int_to_binary(num):
    binary_array = []
    while num > 0:
        binary_array.append(num % 2)
        num = num // 2
    binary_array.reverse()
    while len(binary_array) < 16:
        binary_array.insert(0, 0)
    return binary_array


def prikey_encrypt(Wr, priK):  # 私钥对 Wr 进行签名, 转化为 Cr
    nums = []
    bins = []
    nums = binary_to_int(Wr)
    crypted = RSA_encrypt(priK, nums)
    for ele in crypted:
        elebins = int_to_binary(ele)
        bins += elebins
    return bins     # 返回一维二进制流


def pubkey_decrypt(cr, pubK):  # 公钥解密
    nums = []
    bins = []
    nums = binary_to_int(cr)
    decrypted = RSA_decrypt(pubK, nums)
    for ele in decrypted:
        elebins = int_to_binary(ele)
        bins += elebins
    return bins    # 返回一维二进制流


# 将图像划分为单个小块
def divide_blocks(img, length, height, width):
    cols = int(width/length)
    rows = int(height/length)
    sub_images = []
    for r in range(rows):
        for c in range(cols):
            sub_image = img[r*length:(r+1)*length, c*length:(c+1)*length]
            sub_image = np.expand_dims(sub_image, axis=2)
            sub_images.append(sub_image)
    return rows, cols, sub_images


def H(H_img, W_img, height, width, Xr, block_index):
    # 这里为什么要将 block_index 作为哈希的一部分呢?
    # 它是要确保每个位置的像素块的唯一性,避免恶意复制粘贴 16*16 像素块逃过水印检测
    # 这是原先的设计里有漏洞的一环,我们给它补上了
    # 为什么要将定值 H_img 和 W_img 作为哈希的一部分,这倒很简单,预防攻击者修改图像的尺寸
    combined_array = np.concatenate(([H_img, W_img, block_index], Xr))
    # 计算数组的哈希值
    hash_value = hashlib.sha256(combined_array.tobytes()).digest()
    # 将哈希值转换为二进制字符串
    binary_string = ''.join(format(byte, '08b') for byte in hash_value)
    # 将二进制字符串转换为二维数组
    binary_array = np.array(list(binary_string))
    binary_array = binary_array[:height * width]  # 裁剪为指定长度
#    binary_array = binary_array.reshape((height, width)).astype(int)
    outbins = []
    for i in binary_array.tolist():
        if i == '1':
            outbins.append(1)
        else:
            outbins.append(0)
    return outbins


def mark_block(Xr, priK, Br, N, M, height, width, index):  # 接收二维的 Xr,B
    Xr = Xr.ravel()
    # !一位数组 Xr 进行 LSB置 0
    for i in range(height*width):
        if Xr[i] % 2 == 1:
            Xr[i] -= 1
    # 此时的 Xr 已经是 LSB置 0后的了
    # !计算哈希值
    Pr = H(N, M, height, width, Xr, index)
    Pr = np.array(Pr)

    # !Pr,Br异或得到 Wr
    Br = np.ravel(Br)
    Wr = np.zeros(Xr.shape, np.uint8)
    for i in range(height*width):
        if Pr[i] != (Br[i]//255):
            Wr[i] = 1

    # !私钥加密 Wr成 Cr
    Cr = prikey_encrypt(Wr, priK)
    Cr = np.array(Cr)

    # !Cr 嵌入到 Xr 的 LSB
    for i in range(height*width):
        Xr[i] += Cr[i]
        # 这里的 Xr 实际已是原理图中的 Yr
    Xr = Xr.reshape(height, width)
    # 输出二维数组
    return Xr


def put_watermark(img, priK, B):
    length = 16
    N, M = img.shape[0], img.shape[1]

    # 先对图像和二值比特面裁切为小块,分别置于 sub_imgs,sub_Bs列表里
    row_img, col_img, sub_imgs = divide_blocks(img, length, N, M)
    row_B, col_B, sub_Bs = divide_blocks(B, length, N, M)

    # 此数组存放经水印处理后的小块
    single_blocks = []
    for i in range(len(sub_imgs)):
        # !核心操作 :
        ele = mark_block(sub_imgs[i], priK, sub_Bs[i], N, M, length, length, i)
        single_blocks.append(ele)

    re_img = np.zeros((N, M), dtype=np.uint8)
    for i in range(row_img):
        for j in range(col_img):
            subs = single_blocks[i * col_img + j]
            re_img[i * length:(i+1) * length, j * length:(j+1) * length] = subs

    return re_img


def get_block_mark(Yr, pubK, N, M, height, width, block_index):
    Yr = Yr.ravel()
    # ! 先提取一维数组 Yr的 LSB,记为 Cr
    Cr = np.zeros(height*width, np.uint8)
    for i in range(height*width):
        if Yr[i] % 2 == 1:
            Cr[i] = 1

    # ! 对 Cr 使用公钥解密, 得到 Wr
    Wr = pubkey_decrypt(Cr, pubK)   # Wr为一维二进制流
    Wr = np.array(Wr)
    # ! 一维数组 Yr 进行 LSB置 0
    for i in range(height*width):
        if Yr[i] % 2 == 1:
            Yr[i] -= 1
    # 此时的 Yr 已经是 LSB置 0后的了

    # ! 计算哈希值
    Pr = H(N, M, height, width, Yr, block_index)   # Pr为一维二进制流
    Pr = np.array(Pr)

    # ! Pr,Wr异或得到 Br
    Br = np.zeros(height*width, np.uint8)
    for i in range(height*width):
        if Pr[i] != Wr[i]:
            Br[i] = 255
    Br = Br.reshape(height, width)
    return Br


def extract_watermark(img, pubK):
    length = 16
    N, M = img.shape[0], img.shape[1]

    # 先对图像和二值比特面裁切为小块,分别置于 sub_imgs,sub_Bs列表里
    row_img, col_img, sub_imgs = divide_blocks(img, length, N, M)

    # 此数组存放经水印处理后的小块
    single_blocks = []
    for i in range(len(sub_imgs)):
        # !核心操作 :
        ele = get_block_mark(sub_imgs[i], pubK, N, M, length, length, i)
        single_blocks.append(ele)

    re_img = np.zeros((N, M), dtype=np.uint8)
    for i in range(row_img):
        for j in range(col_img):
            subs = single_blocks[i * col_img + j]
            re_img[i * length:(i+1) * length, j * length:(j+1) * length] = subs

    return re_img


def generate_chessboard(size):  # 创建正方形棋盘状黑白图像
    image = np.zeros((size, size), dtype=np.uint8)
    block_size = size // 64  # 每个小格子的大小
    for y in range(0, size, block_size):
        for x in range(0, size, block_size):
            row = y // block_size
            col = x // block_size
            color = 255 if (row + col) % 2 == 0 else 0
            image[y:y+block_size, x:x+block_size] = color
    return image


def generate_white(size):  # 创建一个大小为 (size, size) 的全白色灰度图像
    image = np.ones((size, size), dtype=np.uint8)*255
    return image


# ------- 以下仅为测试用的函数,这里模拟了一些简单常见的图像篡改方式 --------

def alter_SQ_pixels(image):  # 将左上角 128*128的区域的像素值 +1
    # 获取图像的宽度和高度
    height, width = image.shape[:2]
    # 创建一个副本，用于修改像素值
    modified_image = image.copy()
    for y in range(min(height, 128)):
        for x in range(min(width, 128)):
            if modified_image[y, x] < 255:
                modified_image[y, x] += 1
    return modified_image


def alter_pixels(image):   # 随机将图像的某一个区域的像素值 +1
    height, width = image.shape[:2]
    max_region_area = (height * width) // 5   # 最多影响 20%的面积
    region_width = random.randint(1, min(width, int(np.sqrt(max_region_area))))
    region_height = random.randint(
        1, min(height, int(np.sqrt(max_region_area))))
    # 随机选择区域的左上角坐标
    region_x = random.randint(0, width - region_width)
    region_y = random.randint(0, height - region_height)
    # 创建一个副本，用于修改像素值
    modified = image.copy()
    # 循环遍历区域内的像素
    for y in range(region_y, region_y + region_height):
        for x in range(region_x, region_x + region_width):
            if modified[y, x] < 255:
                modified[y, x] += 1
    return modified


def paste_image(img1, img2, position):  # 将 img1 粘贴到 img2 的某个位置
    width1, height1 = img1.shape[0], img1.shape[1]
    width2, height2 = img2.shape[0], img2.shape[1]

    if position[0] < 0 or position[0] + width1 > width2 or \
            position[1] < 0 or position[1] + height1 > height2:
        print("Error: Invalid position!")
        return
    img2[position[0]:position[0]+width1, position[1]:position[1]+height1] = img1
    return img2


def ctrl_cv(img):  # 将图像的左上角 16*16 的区域复制到相邻的另一个 16*16 区域
    a_region = [0, 0, 16, 16]
    a_block = img[a_region[1]:a_region[3], a_region[0]:a_region[2]]
    b_region = (a_region[2], a_region[1], a_region[2] * 2, a_region[3])
    img[b_region[1]:b_region[3], b_region[0]:b_region[2]] = a_block
    return img


def gaussian(image, mean, std_dev):  # 高斯噪声攻击
    # 生成与图像大小相同的高斯噪声
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
    # 将图像和噪声相加
    noisy_image = cv2.add(image, noise)
    # 确保像素值在有效范围内（0-255）
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def salt_pepper(image, salt_prob, pepper_prob):  # 椒盐噪声攻击
    noisy_image = np.copy(image)
    salt_noise = np.random.rand(image.shape[0], image.shape[1]) < salt_prob
    pepper_noise = np.random.rand(image.shape[0], image.shape[1]) < pepper_prob
    noisy_image[salt_noise] = 255
    noisy_image[pepper_noise] = 0
    return noisy_image


# 计算两幅图像的PSNR值
def psnr_cal(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mse = np.mean((gray1 - gray2) ** 2)
    maxi = np.max(gray1)
    psnr = 10 * np.log10((maxi ** 2) / mse)
    return psnr


# ssim , 从结构, 亮度, 相似度来评判加密效果
def ssim_cal(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 使用解构赋值的方式将 ssim 返回结果分配给变量 ssim_score和下划线。
    # ssim函数返回一个元组，包含 ssim 指标和其他辅助信息，这里忽略辅助信息
    ssim_score, _ = ssim(img1, img2, full=True)
    return ssim_score


# ------------------ 以下是测试代码 ------------------
"""
生成一个 512*512 的全白色灰度图像,这就是所谓的原始"水印"
也可以调用上面的 generate_chessboard 函数生成一个棋盘状的水印
由于"水印"图像的像素点是要和二进制流进行异或的,所以它必须为二值图像
它就必须是灰度图,并且只有 0 和 255 两种像素值 (异或前除以255即可)
你可以自由设计此水印图像,也许还能放置一些信息上去.虽然它的根本目的其实并非隐藏信息
它只是为了可视化地验证图像是否被篡改以及显示出篡改的区域
"""
B = generate_white(512)

# 用作加水印图像尺寸不要太大,建议 512*512,毕竟 python 计算慢的离谱
# 图像长宽需为 16 的倍数, 原始水印也要根据测试图像的大小进行生成

# # 对于灰度图
# img = cv2.imread("./g12/08.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# marked = put_watermark(img, private_key, B)
# modified = ctrl_cv(marked)
# mark_got = extract_watermark(modified, public_key)
# cv2.imshow("after watermark", marked)
# cv2.imshow("modify marked", modified)
# cv2.imshow("extract watermark", mark_got)
# cv2.imshow("original B", B)


# 对于彩色图
img = cv2.imread("./pics/lena.jpg")
cv2.imshow("original", img)
b, g, r = cv2.split(img)
marked_b = put_watermark(b, private_key, B)
marked_g = put_watermark(g, private_key, B)
marked_r = put_watermark(r, private_key, B)
marked = cv2.merge(((marked_b, marked_g, marked_r)))
cv2.imshow("after watermark", marked)
modified_b = alter_pixels(marked_b)
modified_g = alter_pixels(marked_g)
modified_r = alter_pixels(marked_r)
modified = cv2.merge(((modified_b, modified_g, modified_r)))
cv2.imshow("modified watermarked", modified)
b_m, g_m, r_m = cv2.split(modified)
extract_b = extract_watermark(b_m, public_key)
extract_g = extract_watermark(g_m, public_key)
extract_r = extract_watermark(r_m, public_key)
cv2.imshow("extract mark in B", extract_b)
cv2.imshow("extract mark in G", extract_g)
cv2.imshow("extract mark in R", extract_r)


cv2.waitKey(0)
cv2.destroyAllWindows()
