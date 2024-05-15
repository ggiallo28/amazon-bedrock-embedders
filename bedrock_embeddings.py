from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel, model_validator, Field
from datetime import datetime, date
from cat.mad_hatter.mad_hatter import MadHatter
from collections import defaultdict
import boto3
from typing import Optional

mad = MadHatter()


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
            "service_name": self.service_name,
            "region_name": self.service_name,
        }
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        return session.client(**client_kwargs)


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
        "required_bool1": (bool, Field(default=False)),
        "required_bool2": (bool, Field(default=False)),
        "required_int": (int, ...),
        "optional_int": (int, Field(default=69)),
        "required_str": (str, ...),
        "optional_str": (str, Field(default="meow")),
        "required_date": (date, ...),
        "optional_date": (date, Field(default=date.fromtimestamp(1679616000))),
        "aws_region": (str, Field(default="us-east-1")),
    }

    response = client.list_foundation_models(byOutputModality="EMBEDDING")
    models = defaultdict(list)
    for model in response["modelSummaries"]:
        model_name = f"{model['providerName']} {model['modelName']}"
        model_id = model["modelId"]
        models[model_name].append(model_id)
    models = dict(models)

    for model_name, model_ids in models.items():
        fields[model_name] = (list, Field(default_factory=lambda: model_ids))
    dynamic_model = type("DynamicModel", (BaseModel,), fields)
    return dynamic_model


DynamicModel = create_dynamic_model()


class AmazonBedrockEmbeddingsSettings(DynamicModel):

    @model_validator(mode="before")
    def ensure_opposites(cls, values):
        if values.get("required_bool1"):
            values["required_bool2"] = False
        elif values.get("required_bool2"):
            values["required_bool1"] = False
        return values


@plugin
def settings_model():
    return AmazonBedrockEmbeddingsSettings
