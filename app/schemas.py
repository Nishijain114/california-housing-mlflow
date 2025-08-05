from pydantic import BaseModel, conlist
from typing import List

class HousingInput(BaseModel):
    data: List[conlist(float, min_length=8, max_length=8)]
