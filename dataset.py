from prova_lettura_immagini import read_pgm
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\angio\PycharmProjects\AI_Project\DaimlerBenchmark\simple_dataset\1")
path1 = os.getcwd()
non_ped_path = path1 + r"\non-ped_examples\\"
ped_path = path1 + r"\ped_examples\\"

for img_path in os.listdir(ped_path):


# img_path = "03m_11s_013926u.pgm"
# img = read_pgm(test_path + img_path, byteorder='<')
# plt.imshow(img, plt.cm.gray)
# plt.show()

print("")
