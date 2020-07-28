from IPython import get_ipython
from time import sleep

s1 = "/content/clouderizer/melanoma/code/isic2019"
d1 = "/content/drive/My\ Drive/clouderizer/melanoma/isic2019/code"
s2 = "/content/clouderizer/melanoma/out"
d2 = "/content/drive/My\ Drive/clouderizer/melanoma/isic2019/out"

while True:
    get_ipython().system(f"rclone sync {s1} {d1}")
    get_ipython().system(f"rclone sync {s2} {d2}")
    sleep(600)
