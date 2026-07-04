"""
Các hàm gọi từ src: eval_transform, palm_net (kiến trúc model)
Các file được worker gọi: config, dao, locker, model, utils
Các file được server gọi: config, valid, dectect (Mediapipe)

Server <----> Worker: chạy trên process riêng, giao tiếp thông qua các Queue, Server tắt thì Worker tắt

Worker làm việc với database(lưu, so sánh embeddings), esp32 (gửi tín hiệu mở tủ)
Server làm việc với esp32_cam (nhận ảnh, phản hồi lại)
"""

# WORKER
MODEL_NAME = "D:/Projects/Personal/PalmLocker/models/palmnet_arcface_best.pth"
ESP32_IDR = "http://172.20.10.6/open"

SEND_IMAGES = 5 # mỗi lần gửi đồ hệ thống sẽ lấy 5 ảnh lòng bàn tay của người dùng
TAKE_IMAGES = 2 # mỗi lần lấy đồ hệ thống sẽ lấy 2 ảnh lòng bàn tay của người dùng

THRESHOLD = 0.65 # lớn hơn ngưỡng này được xem là cùng 1 người

# SERVER
PORT = 5000 # cổng server chạy
HOST = "0.0.0.0"
STORAGE_PATH = "D:/Projects/Personal/PalmLocker/storage"

# TIMEOUT
TIMEOUT = 15
INVALID_COUNTER = 30

# DATABASE
DATABASE_NAME="palm_lockers"
USER="postgres"
PASSWORD="tienson"
DATABASE_PORT="5432"