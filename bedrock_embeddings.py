from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel, model_validator, Field, create_model, ConfigDict
from datetime import datetime, date
from cat.mad_hatter.mad_hatter import MadHatter
from collections import defaultdict
import boto3
from typing import Any, List, Mapping, Optional, Type
from langchain_community.embeddings import BedrockEmbeddings
from cat.factory.embedder import EmbedderSettings

mad = MadHatter()

DEFAULT_MODEL =  "amazon.titan-embed-text-v1"
DEFAULT_REGION = "us-east-1"


class Boto3ClientBuilder:
    def __init__(
        self,
        service_name: str,
        region_name: str,
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        iam_role_assigned: Optional[bool] = False,
    ):
        self.service_name = service_name
        self.profile_name = profile_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url
        self.iam_role_assigned = iam_role_assigned
        self.region_name = region_name

    def set_profile_name(self, profile_name: str):
        self.profile_name = profile_name

    def set_credentials(self, aws_access_key_id: str, aws_secret_access_key: str):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

    def set_endpoint_url(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def build_client(self):
        if self.iam_role_assigned:
            session = boto3.Session()
        elif self.profile_name:
            session = boto3.Session(profile_name=self.profile_name)
        else:
            session_kwargs = {}
            if self.aws_access_key_id and self.aws_secret_access_key:
                session_kwargs["aws_access_key_id"] = self.aws_access_key_id
                session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
            session = boto3.Session(**session_kwargs)
        client_kwargs = {
            "region_name": self.region_name,
        }
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        return session.client(self.service_name, **client_kwargs)


aws_plugin_name = next(
    (name for name in mad.active_plugins if name.startswith("aws_integration")), None
)

if aws_plugin_name:
    aws_integration = mad.plugins.get(aws_plugin_name)
    if aws_integration:
        settings = aws_integration.load_settings()
        client_builder = Boto3ClientBuilder(
            service_name="bedrock",
            profile_name=settings.get("credentials_profile_name"),
            aws_access_key_id=settings.get("aws_access_key_id"),
            aws_secret_access_key=settings.get("aws_secret_access_key"),
            endpoint_url=settings.get("endpoint_url"),
            iam_role_assigned=settings.get("iam_role_assigned"),
            region_name=settings.get("region_name"),
        )
        client = client_builder.build_client()
        print("AWS client successfully created.")
    else:
        print("AWS integration plugin found, but could not be accessed.")
else:
    print("No AWS integration plugin found.")


def create_dynamic_model() -> BaseModel:
    fields = {
        "model_id": (
            str,
            Field(default=DEFAULT_MODEL, description="Unique identifier for the model."),
        ),
        "model_kwargs": (
            dict,
            Field(
                default={},
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

    fields = {**fields, **dynamic_fields}
    dynamic_model = create_model("DynamicModel", **fields)

    class AmazonBedrockEmbeddingsSettings(dynamic_model):
        @model_validator(mode="before")
        def ensure_opposites(cls, values):
            true_fields = [
                field
                for field in dynamic_fields.keys()
                if values.get(field, False)
            ]

            if true_fields:
                values["model_id"] = models[true_fields[0]]
                for field in true_fields[1:]:
                    values[field] = False
            else:
                values["model_id"] = values.get("model_id", DEFAULT_MODEL)
                    
            return values

    return AmazonBedrockEmbeddingsSettings


@plugin
def settings_model():
    return create_dynamic_model()


class CustomBedrockEmbeddings(BedrockEmbeddings):
    def __init__(self, **kwargs: Any) -> None:
        input_kwargs = {
            "model_id": kwarge.get("model_id", DEFAULT_MODEL),
            "credentials_profile_name": settings.get("credentials_profile_name"),
            "endpoint_url": settings.get("endpoint_url"),
            "normalize": kwargs.get("normalize", False),
            "model_kwargs": kwargs.get("model_kwargs"),
            "client": client
        }
        input_kwargs = {k: v for k, v in input_kwargs.items() if v is not None}
        super().__init__(**input_kwargs)

class AmazonBedrockEmbeddingsConfig(EmbedderSettings):
    model_id: str = DEFAULT_MODEL
    model_kwargs: dict = {}
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
