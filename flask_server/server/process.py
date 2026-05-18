import os
import time
from datetime import datetime

import cv2
import numpy as np

from flask_server.common.config import TIMEOUT, SEND_IMAGES, TAKE_IMAGES, INVALID_COUNTER, STORAGE_PATH

from flask_server.server.detect import detect_hand
from flask_server.server.queues import queues
from flask_server.server.valid import is_palm_open, is_palm_large_enough, crop_palm_roi
from flask_server.server.state import state
from flask_server.server.ws import ws_manager

def _start_new_session():
    now = datetime.now()

    session_path = os.path.join(
        STORAGE_PATH,
        now.strftime("%Y"),
        now.strftime("%m"),
        now.strftime("%d"),
        now.strftime("%H_%M_%S_%f")[:-3]
    )
    session_id = now.strftime("%Y%m%d_%H%M%S")

    raw_path = os.path.join(session_path, "raw")
    roi_path = os.path.join(session_path, "roi")

    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(roi_path, exist_ok=True)

    state.set_current_session_dir(session_path)

    return session_id, raw_path, roi_path

def _save_images(raw_images, roi_images, raw_path, roi_path):
    for i, (raw, roi_item) in enumerate(zip(raw_images, roi_images)):
        roi = roi_item["image"]

        raw_file = os.path.join(raw_path, f"{i:03d}.jpg")
        roi_file = os.path.join(roi_path, f"{i:03d}.jpg")

        cv2.imwrite(raw_file, raw)
        cv2.imwrite(roi_file, roi)

def preprocess():
    invalid_counter = 0
    valid_counter = 0
    raw_images = []
    roi_images = []
    while True:
        if state.get_reset_flag():
            raw_images.clear()
            roi_images.clear()
            invalid_counter = 0
            valid_counter = 0
            state.set_reset_flag(False)
            state.set_mode(None)
            state.set_start_time(None)
            continue

        jpeg = queues.get_frame()

        mode = state.get_mode()
        print(f"[PROCESS]: Received mode: {mode}")
        if mode is None:
            continue
        # print(f"[PROCESS] Queues size:{queues.size()}")
        s = time.time()
        np_arr = np.frombuffer(jpeg, np.uint8) # convert bytes -> numpy array
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # decode jpeg -> BGR image
        if frame is None:
            continue
        ok, msg1, hand = detect_hand(frame) #ktra là bàn tay
        # Time vượt quá TIMEOUT mà chưa phát hiện được bàn tay nào hợp lệ thì dừng
        current_time = time.time()
        start_time = state.get_start_time()
        if start_time is not None and (current_time - start_time) >= TIMEOUT:
            if invalid_counter > INVALID_COUNTER:
                if ws_manager.get_connection() is not None:
                    ws_manager.send("fail")
                    print("[PROCESS] Sent: fail")
                raw_images.clear()
                roi_images.clear()
                invalid_counter = 0
                valid_counter = 0
                state.set_mode(None)
                state.set_start_time(None)
                queues.clear_frame()
                continue

        if not ok:
            invalid_counter += 1
            continue

        is_open, msg2 = is_palm_open(hand) # ktra tay mở
        if not is_open:
            invalid_counter += 1
            if ws_manager.get_connection() is not None:
                ws_manager.send("close")
                print("[PROCESS] Sent: close")
            continue

        is_enough, msg3 = is_palm_large_enough(hand, frame.shape) # ktra tay đủ lớn
        if not is_enough:
            invalid_counter += 1
            if ws_manager.get_connection() is not None:
                ws_manager.send("small")
                print("[PROCESS] Sent: small")
            continue

        raw_images.append(frame)

        if ws_manager.get_connection() is not None:
            ws_manager.send("valid")
            print("[PROCESS] Sent: valid")

        roi = crop_palm_roi(frame, hand, roi_size=224)
        print(f"[Latency] Latency on preprocessing: {time.time() - s}")

        if roi_images and roi_images[-1]["mode"] != mode:
            roi_images.clear()
            raw_images.clear()
            valid_counter = 0

        roi_images.append({
            "image": roi,
            "mode": mode
        })

        valid_counter += 1
        print(f"[PROCESS]: received {valid_counter} frames")

        if mode == "take":
            if len(roi_images) >= TAKE_IMAGES:
                if ws_manager.get_connection() is not None:
                    ws_manager.send("wait")
                    queues.put_image({
                        "images": [x["image"] for x in roi_images],
                        "mode": mode
                    })
                roi_images.clear()
                raw_images.clear()
                state.set_mode(None)
                valid_counter = 0
                print("[bold magenta][PROCESS][/bold magenta] Completed")

        elif mode == "send":
            if len(roi_images) >= SEND_IMAGES:
                if ws_manager.get_connection() is not None:
                    ws_manager.send("wait")
                    session_id, raw_path, roi_path = _start_new_session()
                    _save_images(raw_images, roi_images, raw_path, roi_path)
                    queues.put_image({
                        "images": [x["image"] for x in roi_images],
                        "mode": mode,
                        "session_id": session_id
                    })
                roi_images.clear()
                raw_images.clear()
                state.set_mode(None)
                valid_counter = 0
                print(f"[bold magenta][PROCESS][/bold magenta] Completed")

        print(f"Queues: {queues.size()}")