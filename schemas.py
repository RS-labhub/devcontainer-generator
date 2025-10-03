# schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Any

class DevContainerModel(BaseModel):
    name: str = Field(description="Name of the dev container")
    image: str = Field(description="Docker image to use")
    forwardPorts: Optional[list[int]] = Field(
        default=None,
        description="Ports to forward from the container to the local machine"
    )
    customizations: Optional[dict[str, Any]] = Field(
        default=None, 
        description="Tool-specific configuration. Use proper nested structure with matching braces."
    )
    settings: Optional[dict[str, Any]] = Field(
        default=None, 
        description="VS Code settings to configure the development environment"
    )
    postCreateCommand: Optional[str] = Field(
        default=None,
        description="Command to run after creating the container"
    )
    
    class Config:
        # Allow extra fields that might be in devcontainer.json
        extra = "allow"