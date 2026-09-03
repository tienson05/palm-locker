import threading
import time
from multiprocessing import Process

from flask import Flask, request
from flask_sock import Sock

from server.common.config import HOST, PORT
from server.common.dao import dao

from server.app.process import preprocess
from server.app.ws import ws_manager
from server.app.queues import queues
from server.app.state import state

from server.worker.worker import worker as worker_fn

app = Flask(__name__)
sock = Sock(app)

def ws_sender():
    while True:
        data = queues.get_ws_message()
        ws = ws_manager.get_connection()

        if ws and data is not None:
            try:
                ws_manager.send(str(data))
            except Exception:
                ws_manager.clear(ws)

        print(f"[Server -> ESP32CAM] Sent: {data}")

@app.route("/event", methods=["GET"])
def http_command():
    cmd = request.args.get("command").strip().lower()

    if not cmd:
        return {"error": "Missing command"}, 400

    if ws_manager.get_connection() is None:
        return {"error": "ESP32CAM not connected"}, 503

    if cmd == "send":
        lockers = dao.get_available_locker()
        if not lockers:
            ws_manager.send("full")
            return {"locker": None}, 400

    if cmd == "take":
        lockers = dao.get_active_session()
        if not lockers:
            ws_manager.send("fail")
            return {"locker": None}, 400

    ws_manager.send(cmd) # gửi xuống esp32_cam

    state.set_mode(cmd)
    state.set_start_time(time.time())

    print(f"[Server → ESP32CAM] Sent: {cmd}")
    return {"status": "ok"}

@sock.route("/ws")
def esp32_socket(ws):
    ws_manager.set(ws)

    print(f"[SERVER] ESP32 CAM connected (new)")

    try:
        while True:
            data = ws.receive()
            if data is None:
                break

            if isinstance(data, str):
                print(f"[ESP32CAM → TEXT] {data}")
                continue

            if isinstance(data, bytes):
                jpeg = data

                queues.put_frame(jpeg)
                print(f"[SERVER] Queues size:{queues.size()}")
                print(f"[SERVER] Received JPEG: {len(jpeg)} bytes")
            else:
                print(f"[SERVER] Unknown data type: {type(data)}")

    except Exception as e:
        print(f"[SERVER] Error in receive loop: {e}")
    finally:
        ws_manager.clear(ws)
        print("[SERVER] ESP32 CAM disconnected")
        queues.clear_frame()
        state.set_reset_flag(True)
        state.set_mode(None)
        state.set_start_time(None)

if __name__ == '__main__':
    dao.connect_database()

    threading.Thread(target=preprocess, daemon=True).start()
    threading.Thread(target=ws_sender, daemon=True).start()

    worker = Process(target=worker_fn, args=(queues.image_queue, queues.ws_queue,))
    worker.daemon = True
    worker.start()

    app.run(host=HOST, port=PORT)