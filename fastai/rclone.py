from time import sleep
from subprocess import run

s1 = "/content/clouderizer/melanoma/code"
d1 = "/content/drive/My\ Drive/clouderizer/melanoma/code"
s2 = "/content/clouderizer/melanoma/out"
d2 = "/content/drive/My\ Drive/clouderizer/melanoma/out"

while True:
    run(
        [
            "rclone",
            "sync",
            r"/content/clouderizer/melanoma/code",
            r"/content/drive/My\ Drive/clouderizer/melanoma/code",
        ]
    )
    # !rclone sync {s2} {d2}
    break
    sleep(600)
