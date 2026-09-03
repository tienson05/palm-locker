class AppState:
    def __init__(self):
        self.mode = None
        self.current_session_dir = None
        self.start_time = None
        self.reset_flag = False

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode

    def set_current_session_dir(self, current_session_dir):
        self.current_session_dir = current_session_dir

    def get_current_session_dir(self):
        return self.current_session_dir

    def set_start_time(self, start_time):
        self.start_time = start_time

    def get_start_time(self):
        return self.start_time

    def set_reset_flag(self, reset_flag):
        self.reset_flag = reset_flag

    def get_reset_flag(self):
        return self.reset_flag

state = AppState()