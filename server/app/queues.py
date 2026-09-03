from multiprocessing import Queue
import queue

class QueueManager:
    def __init__(self, ws_size=15, frame_size=50, image_size=15):
        self.ws_queue = Queue(maxsize=ws_size)
        self.frame_queue = Queue(maxsize=frame_size)
        self.image_queue = Queue(maxsize=image_size)

    def clear_queue(self, q):
        """Clear toàn bộ queue (non-blocking)"""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def clear_all(self):
        self.clear_queue(self.ws_queue)
        self.clear_queue(self.frame_queue)
        self.clear_queue(self.image_queue)

    def clear_frame(self):
        self.clear_queue(self.frame_queue)

    def safe_put(self, q, item):
        """Put không block, full thì bỏ item cũ"""
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()  # bỏ item cũ
            except queue.Empty:
                pass
            q.put_nowait(item)

    # PUT
    def put_ws_message(self, item):
        self.safe_put(self.ws_queue, item)

    def put_frame(self, item):
        self.safe_put(self.frame_queue, item)

    def put_image(self, item):
        self.safe_put(self.image_queue, item)

    # GET
    def get_ws_message(self):
        return self.ws_queue.get()

    def get_frame(self):
        return self.frame_queue.get()

    def get_image(self):
        return self.image_queue.get()

    # DEBUG
    def size(self):
        return {
            "ws": self.ws_queue.qsize(),
            "frame": self.frame_queue.qsize(),
            "image": self.image_queue.qsize()
        }

queues = QueueManager()