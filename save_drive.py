from IPython import get_ipython
from time import sleep

s1 = "/content/clouderizer/melanoma/code"
d1 = "/content/drive/My\ Drive/clouderizer/melanoma/code"
s2 = "/content/clouderizer/melanoma/out/2020.test_effb0_rr_oc"
d2 = "/content/drive/My\ Drive/clouderizer/melanoma/out/2020.test_effb0_rr_oc"

while True:
    #get_ipython().system(f"rclone sync {s1} {d1} -q --fast-list")
    get_ipython().system(f"rclone sync {s2} {d2} --fast-list -v")
    sleep(360)
