from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    message: str = Field(max_length=5_000, examples=["You won a free prize. Call now!"])

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class PredictionResponse(BaseModel):
    label: str
