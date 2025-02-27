# Amazon Bedrock Embedders

[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=383938&style=for-the-badge&logo=cheshire_cat_ai)](https://)  
[![Awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=Awesome+plugin&color=000000&style=for-the-badge&logo=cheshire_cat_ai)](https://)  
[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=F4F4F5&style=for-the-badge&logo=cheshire_cat_black)](https://)

This plugin integrates Amazon Bedrock Embeddings into the Cheshire Cat AI framework, allowing you to use various Amazon Bedrock models for text embedding.

## Features

- Dynamic loading of available Amazon Bedrock embedding models
- Customizable settings for each embedding model
- Integration with the Cheshire Cat AI plugin system
- Support for multiple embedding models in a single plugin

## Usage

1. Ensure you have the necessary AWS credentials and permissions to access Amazon Bedrock services.
2. Install the plugin in your Cheshire Cat AI environment.
3. Configure the plugin settings to enable/disable specific Amazon Bedrock embedding models.
4. The plugin will automatically integrate the selected embedding models into the Cheshire Cat AI framework.

## Configuration

The plugin dynamically generates settings based on the available Amazon Bedrock embedding models. You can enable or disable specific models through the plugin settings.

Default model: `amazon.titan-embed-text-v1`

## Requirements

- AWS account with access to Amazon Bedrock services
- `boto3` Python library
- Cheshire Cat AI framework

## Development

To modify or extend this plugin:

1. Update the `bedrock_embedders.py` file to add new features or modify existing functionality.
2. Adjust the `get_availale_models` function if you need to change how available models are fetched.
3. Modify the `create_custom_bedrock_class` function to customize the behavior of individual embedding models.

Remember to update the `version` in the `plugin.json` file when making changes to trigger a new release.

## License

[Add your license information here]
