class StateManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(StateManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_state"):
            self._state = {}

    def set_company_and_user(self, user_id: int, user_query: str):
        self._state["user_id"] = user_id
        self._state["user_query"] = user_query

    def get_user_id(self) -> int:
        return self._state.get("user_id", None)
    
    def get_user_query(self) -> int:
        return self._state.get("user_query", None)

    def get_user_details(self) -> dict:
        return self._state
