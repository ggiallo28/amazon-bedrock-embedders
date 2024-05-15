from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel, model_validator, Field, create_model, ConfigDict
from datetime import datetime, date
from cat.mad_hatter.mad_hatter import MadHatter
from collections import defaultdict
import boto3
from typing import Any, List, Mapping, Optional, Type
from langchain_community.embeddings import BedrockEmbeddings
from cat.factory.embedder import EmbedderSettings
from cat.log import log
import json

mad = MadHatter()

DEFAULT_MODEL =  "amazon.titan-embed-text-v1"

def get_plugin_by_name(mad_hatter, prefix):
    """Retrieve the first plugin with a name starting with 'prefix'."""
    for name in mad_hatter.active_plugins:
        if name.startswith(prefix):
            return mad_hatter.plugins.get(name)
    return None

def create_aws_client(plugin, service_name):
    """Create an AWS client using settings from the plugin."""
    try:
        settings = plugin.load_settings()
        if not settings:
            log.error("Failed to load settings from the plugin.")
            return None
            
        aws_model = plugin.settings_model()
        if not aws_model:
            log.error("No settings model available in the plugin.")
            return None

        client = aws_model.get_aws_client(settings, service_name=service_name)
        log.info("AWS client successfully created.")
        return client
    except Exception as e:
        log.error("An error occurred while creating the AWS client: %s", e)
        return None

def get_aws_client(service_name="bedrock"):
    """Retrieve an AWS client for a specific service using a plugin."""
    aws_plugin = get_plugin_by_name(mad, "aws_integration")
    if aws_plugin:
        client = create_aws_client(aws_plugin, service_name=service_name)
        if client:
            return client
        else:
            logger.info("AWS integration plugin found, but the client could not be accessed or created.")
    else:
        logger.info("No AWS integration plugin found.")

def create_dynamic_model(client) -> BaseModel:
    response = client.list_foundation_models(byOutputModality="EMBEDDING")
    models = defaultdict(list)
    for model in response["modelSummaries"]:
        modelName = f"{model['providerName']} {model['modelName']}"
        modelId = model["modelId"]
        models[modelName].append(modelId)

    dynamic_fields = {}
    for modelName, model_ids in models.items():
        dynamic_fields[modelName] = (
            bool,
            Field(default=False, description=f"Enable/disable the {modelName} model."),
        )
    
    fields = {
        "model_id": (
            str,
            Field(
                default=DEFAULT_MODEL, description="Unique identifier for the model."
            ),
        ),
        "model_kwargs": (
            str,
            Field(
                default="{}",
                description="Keyword arguments specific to the model configuration.",
            ),
        ),
        "normalize": (
            bool,
            Field(
                default=False,
                description="Flag to determine if the model output should be normalized.",
            ),
        ),
    }

    fields = {**fields, **dynamic_fields}
    dynamic_model = create_model("DynamicModel", **fields)

    class AmazonBedrockEmbeddingsSettings(dynamic_model):
        
        @model_validator(mode="before")
        def ensure_opposites(cls, values):
            true_fields = [
                field for field in dynamic_fields.keys() if values.get(field, False)
            ]

            if true_fields:
                values["model_id"] = models[true_fields[0]][0]
                for field in true_fields[1:]:
                    values[field] = False
            else:
                values["model_id"] = values.get("model_id", DEFAULT_MODEL)

            return values

    return AmazonBedrockEmbeddingsSettings

client = get_aws_client()

@plugin
def settings_model():
    return create_dynamic_model(client)

class CustomBedrockEmbeddings(BedrockEmbeddings):
    def __init__(self, **kwargs: Any) -> None:
        input_kwargs = {
            "model_id": kwargs.get("model_id", DEFAULT_MODEL),
            "normalize": kwargs.get("normalize", False),
            "model_kwargs": json.loads(kwargs.get("model_kwargs")),
            "client": client,
        }
        input_kwargs = {k: v for k, v in input_kwargs.items() if v is not None}
        super().__init__(**input_kwargs)


class AmazonBedrockEmbeddingsConfig(EmbedderSettings):
    model_id: str = DEFAULT_MODEL
    model_kwargs: str = "{}"
    normalize: bool = False
    _pyclass: Type = CustomBedrockEmbeddings

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Amazon Bedrock Embeddings",
            "description": "Configuration for Amazon Bedrock Embeddings",
            "link": "https://aws.amazon.com/bedrock/",
        }
    )


@hook
def factory_allowed_embedders(allowed, cat) -> List:
    global plugin_path
    plugin_path = mad.get_plugin().path
    allowed.append(AmazonBedrockEmbeddingsConfig)
    return allowed
