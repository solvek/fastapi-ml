from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "SolvekNLP server"
    API_V1_STR: str = "/api/v1"

    MODEL_PATH: str = "app/ml_model/model.pth"

    class Config:
        case_sensitive = True


settings = Settings()
