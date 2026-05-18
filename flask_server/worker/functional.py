import json
import numpy as np

from flask_server.common.config import THRESHOLD

def compare_embeddings(query_embedding, dao):
    mean_embeddings = dao.get_active_session()

    if not mean_embeddings:
        return None, -1

    # load toàn bộ embedding
    embeddings = []
    locker_ids = []

    for item in mean_embeddings:
        embeddings.append(json.loads(item[2]))
        locker_ids.append(item[4])

    embeddings = np.array(embeddings, dtype=np.float32)  # (N, D)

    # cosine similarity vì đã normalize → chỉ cần dot
    scores = embeddings @ query_embedding  # (N,)

    best_idx = np.argmax(scores)
    best_score = float(scores[best_idx])
    best_locker = locker_ids[best_idx]

    if best_score < THRESHOLD:
        return None, best_score

    return best_locker, best_score

def save_to_db(session_id, mean_embedding, dao):
    lockers = dao.get_available_locker()
    print("[WORKER] Available lockers: ", lockers)
    if not lockers:
        return None

    locker_id = lockers[0][0]
    embedding = mean_embedding.astype(np.float32).tolist()
    dao.add_session(session_id, locker_id, palm_hash=json.dumps(embedding))
    print("[WORKER] Saved locker: ", locker_id)
    return locker_id