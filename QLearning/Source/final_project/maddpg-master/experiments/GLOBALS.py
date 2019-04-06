class GLOBALS:
    predator = 2
    prey = 1
    state = []

    @staticmethod
    def get_vars():
        global g_vars
        return g_vars

    @staticmethod
    def get_state():
        global state
        return state

    @staticmethod
    def append_state(a):
        global state
        state.append(a)

    @staticmethod
    def reset_state():
        global state
        state = []


global_env = GLOBALS()
