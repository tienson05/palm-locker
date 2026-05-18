from flask_server.common.dao import dao
from flask_server.worker.locker import send_locker
from flask_server.worker.functional import compare_embeddings, save_to_db
from flask_server.worker.model import load_model, get_mean_embedding
from rich import print

def worker(image_queue, ws_queue):
    print("[WORKER] Starting...")
    dao.connect_database()
    model, device = load_model()
    print("[WORKER] Ready")

    while True:
        try:
            data = image_queue.get()

            images = data.get("images")
            mode = data.get("mode")

            if not images or mode is None:
                ws_queue.put("fail")
                continue

            mean_embedding = get_mean_embedding(model, device, images) # trả về mean của batch luôn

            if mode == "take":
                lock_id, best_score = compare_embeddings(mean_embedding, dao)
                if lock_id is not None:
                    ok = send_locker(lock_id)
                    if ok:
                        dao.deactivate_active_sessions(lock_id)
                        ws_queue.put("done")
                    else:
                        ws_queue.put("fail")
                else:
                    ws_queue.put("fail")
                print(f"[WORKER] Locker, Best_score: {lock_id}, {best_score}")

            elif mode == "send":
                session_id = data.get("session_id")
                lock_id = save_to_db(session_id, mean_embedding, dao)
                ok = send_locker(lock_id)
                if ok:
                    ws_queue.put(int(lock_id))
                else:
                    ws_queue.put("fail")
                print(f"[WORKER] Lock: {lock_id}")
        except Exception as e:
            print(f"[WORKER] Erron {e}")
            ws_queue.put("fail")