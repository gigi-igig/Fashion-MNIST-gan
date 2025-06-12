# config.py
import os

# 自動偵測執行環境（預設為 kaggle）
RUN_ENV = os.getenv("RUN_ENV", "kaggle")

# 輸出資料夾
if RUN_ENV == "local":
    OUTPUT_DIR = "./outputs"
else:
    OUTPUT_DIR = "/kaggle/working"

# 訓練參數
BUFFER_SIZE = 60000
BATCH_SIZE = 128
EPOCHS = 50
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 10
SAVE_INTERVAL = 5  # 每幾個 epoch 儲存一次模型與圖片

# 儲存檔案路徑模板

IMAGE_SAVE_PATH = f"{OUTPUT_DIR}/image_epoch_{{:03d}}.png"
LOG_SAVE_PATH = f"{OUTPUT_DIR}/training_log.csv"
OUTPUT_DIR = "/kaggle/working"
MODEL_SAVE_WEIGHT_PATH = f"{OUTPUT_DIR}/generator_weights_epoch_{{:03d}}.weights.h5"
DISC_SAVE_WEIGHT_PATH = f"{OUTPUT_DIR}/discriminator_weights_epoch_{{:03d}}.weights.h5"
#MODEL_SAVE_PATH = f"{OUTPUT_DIR}/generator_epoch_{{:03d}}.keras"
#DISC_SAVE_PATH = f"{OUTPUT_DIR}/discriminator_epoch_{{:03d}}.keras"