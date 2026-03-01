import os

# 修复 OpenMP 冲突错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from scripts.sb3.render_rollout import main


if __name__ == "__main__":
    main()
