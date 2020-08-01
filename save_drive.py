from IPython import get_ipython
from time import sleep

s1 = "/content/clouderizer/melanoma/code"
d1 = "/content/drive/My\ Drive/clouderizer/melanoma/code"
s2 = "/content/clouderizer/melanoma-2/out/2020.test_effb0_ss"
d2 = "/content/drive/My\ Drive/clouderizer/melanoma/out/2020.test_effb0_ss"

while True:
    #get_ipython().system(f"rclone sync {s1} {d1} -q --fast-list")
    get_ipython().system(f"rclone sync {s2} {d2} --fast-list -v")
    sleep(360)
