from IPython import get_ipython
from time import sleep
from pathlib import Path

BASE_PATH = Path.cwd().expanduser().parent
DRIVE = Path("/content/drive/My\ Drive/clouderizer/melanoma")

s1 = BASE_PATH / "code"
d1 = DRIVE / "code"
s2 = BASE_PATH / "out"
d2 = DRIVE /  "out"

while True:
    #get_ipython().system(f"rclone sync {s1} {d1} -q --fast-list")
    command = f'rclone copy {s2} {d2} --fast-list -v'
    get_ipython().system(command)
    sleep(360)
