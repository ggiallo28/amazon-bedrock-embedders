from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel, model_validator, Field
from datetime import datetime, date
from cat.mad_hatter.mad_hatter import MadHatter
from collections import defaultdict
import boto3

mad = MadHatter()

aws_plugin_name = next((name for name in mad.active_plugins if name.startswith("aws_integration")), None)

if aws_plugin_name:
    aws_integration = mad.plugins.get(aws_plugin_name)
    if aws_integration:
        settings = aws_integration.load_settings()
        print(settings)
    else:
        print("AWS integration plugin found, but could not be accessed.")
else:
    print("No AWS integration plugin found.")
    

def create_dynamic_model() -> BaseModel:
    fields = {
        'required_bool1': (bool, Field(default=False)),
        'required_bool2': (bool, Field(default=False)),
        'required_int': (int, ...),
        'optional_int': (int, Field(default=69)),
        'required_str': (str, ...),
        'optional_str': (str, Field(default="meow")),
        'required_date': (date, ...),
        'optional_date': (date, Field(default=date.fromtimestamp(1679616000))),
        'aws_region': (str, Field(default='us-east-1'))
    }

    client = boto3.client('bedrock')
    response = client.list_foundation_models(
        byOutputModality='EMBEDDING'
    )
    models = defaultdict(list)
    for model in response['modelSummaries']:
        model_name = f"{model['providerName']} {model['modelName']}"
        model_id = model['modelId']
        models[model_name].append(model_id)
    models = dict(models)

    for model_name, model_ids in models.items():
        fields[model_name] = (list, Field(default_factory=lambda: model_ids))
    dynamic_model = type('DynamicModel', (BaseModel,), fields)
    return dynamic_model
    
DynamicModel = create_dynamic_model()

class AmazonBedrockEmbeddingsSettings(DynamicModel):
    
    @model_validator(mode='before')
    def ensure_opposites(cls, values):
        if values.get('required_bool1'):
            values['required_bool2'] = False
        elif values.get('required_bool2'):
            values['required_bool1'] = False
        return values

@plugin
def settings_model():
    return AmazonBedrockEmbeddingsSettings