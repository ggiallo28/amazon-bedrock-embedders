from cat.mad_hatter.decorators import tool, hook, plugin
from cat.factory.embedder import EmbedderSettings
from cat.plugins.aws_integration import Boto3
from pydantic import BaseModel, model_validator, Field, create_model, ConfigDict
from typing import Any, List, Mapping, Optional, Type, Literal
from langchain_community.embeddings import BedrockEmbeddings
from cat.mad_hatter.mad_hatter import MadHatter
from datetime import datetime, date
from collections import defaultdict
from enum import Enum
from cat.log import log
import random
import json
from enum import Enum

    
DEFAULT_MODEL =  "amazon.titan-embed-text-v1"

client = Boto3().get_client("bedrock")

def get_availale_models(client):
    response = client.list_foundation_models(
        byOutputModality="EMBEDDING",
         byInferenceType='ON_DEMAND'
    )
    models = defaultdict(list)
    for model in response["modelSummaries"]:
        modelName = f"{model['providerName']} {model['modelName']}"
        modelId = model["modelId"]
        models[modelName].append(modelId)
    return dict(models)

def create_custom_bedrock_class(model_name, embedder_info):
    class CustomBedrockEmbeddings(BedrockEmbeddings):
        def __init__(self, **kwargs):
            input_kwargs = {
                "model_id": embedder_info[0],
                "normalize": kwargs.get("normalize", False),
                "model_kwargs": json.loads(kwargs.get("model_kwargs", "{}")),
                "client": Boto3().get_client("bedrock-runtime")
            }
            input_kwargs = {k: v for k, v in input_kwargs.items() if v is not None}
            super(CustomBedrockEmbeddings, self).__init__(**input_kwargs)
            
    class_name = model_name.lower().replace(" ", "_")
    CustomBedrockEmbeddings.__name__ = f"CustomBedrockEmbeddings_{class_name}"
    return CustomBedrockEmbeddings

amazon_embedders = get_availale_models(client)
config_embedders = {}
for model_name, embedder_info in amazon_embedders.items():
    custom_bedrock_class = create_custom_bedrock_class(model_name, embedder_info)
    class AmazonBedrockEmbeddingsConfig(EmbedderSettings):
        model_id: str = embedder_info[0]
        model_kwargs: str = "{}"
        normalize: bool = False
        _pyclass: Type = custom_bedrock_class
        
        model_config = ConfigDict(
            json_schema_extra={
                "humanReadableName":  f"Amazon Bedrock: {model_name}",
                "description": "Configuration for Amazon Bedrock Embeddings",
                "link": "https://aws.amazon.com/bedrock/",
            }
        )
    
    new_class = type(model_name, (AmazonBedrockEmbeddingsConfig,), {})
    locals()[model_name] = new_class
    config_embedders[model_name] = new_class
    
def create_dynamic_model(amazon_embedders)-> BaseModel:
    dynamic_fields = {}
    for model_name, model_ids in amazon_embedders.items():
        dynamic_fields[model_name] = (
            bool,
            Field(
                default=model_ids[0] == DEFAULT_MODEL, 
                description=f"Enable/disable the {model_name} model."
            ),
        )
    dynamic_model = create_model("DynamicModel", **dynamic_fields)
    return dynamic_model

DynamicModel = create_dynamic_model(amazon_embedders)
class AmazonBedrockEmbeddingsSettings(DynamicModel):
    
    @classmethod
    def init_embedder(cls):
        if not hasattr(cls, '_current_embedders'):
            setattr(cls, '_current_embedders', [])
            
    @classmethod
    def get_embedders(cls):
        return cls._current_embedders
        
    @model_validator(mode="before")
    def validate(cls, values):
        cls._current_embedders = []
        for emb in values.keys():
            if values[emb]:
                cls._current_embedders.append(config_embedders[emb])
        print("Dynamically Selected:", cls._current_embedders)
        return values

@plugin
def settings_model():
    return AmazonBedrockEmbeddingsSettings
    
@hook
def factory_allowed_embedders(allowed, cat) -> List:
    AmazonBedrockEmbeddingsSettings.init_embedder()
    aws_plugin = MadHatter().plugins.get("bedrock_embedder")
    plugin_settings = aws_plugin.load_settings()
    AmazonBedrockEmbeddingsSettings(**plugin_settings)
    return allowed + AmazonBedrockEmbeddingsSettings.get_embedders()