# LLM Image Content Understanding Chatbot

An advanced chatbot that combines OpenAI's Language Models (LLM) with state-of-the-art image processing techniques. This chatbot understands the content of uploaded images, providing a platform for users to search for relevant images based on textual descriptions, generate meaningful captions, and detect objects within the images.

## Features

- **Image Content Understanding**: Extracts the essence of an image and presents it as a human-readable caption.
- **Textual Image Search**: Allows users to search for images in the database by describing their content, returning images that align with the textual query.
- **Object Detection**: Accurately identifies and locates objects within uploaded images, providing bounding box coordinates.
- **Dynamic Image Database**: Users can continually enrich the image database by uploading new content.

## Usage

1. Engage with the chatbot through the interactive UI.
2. Upload images to enrich the database and to leverage the chatbot's capabilities of understanding, captioning, and object detection.
3. Search for images by describing their content, and the chatbot will retrieve the most relevant images from the database.

## Setup API Keys

Before using the application, you need to set up your OpenAI and HuggingFace API keys. These keys enable the chatbot to access the necessary language models and other resources.

1. Obtain your OpenAI API key from [OpenAI's platform](https://beta.openai.com/signup/).
2. Obtain your HuggingFace API token from the [HuggingFace website](https://huggingface.co/).

Once you have both keys, you can set them in your environment:

### For bash users (e.g., .bashrc):
```bash
echo 'export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"' >> ~/.bashrc
echo 'export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"' >> ~/.bashrc
source ~/.bashrc
```

Replace `YOUR_OPENAI_API_KEY` and `YOUR_HUGGINGFACE_TOKEN` with your actual keys. Ensure you keep your keys confidential and do not commit them to public repositories.

## Running with Docker

You can also run the application using Docker, which provides a consistent and isolated environment. Here's how:

1. Make sure you have Docker installed on your system. If not, you can [download and install Docker](https://www.docker.com/products/docker-desktop).

2. Clone the repository to your local machine:
```bash
git clone https://github.com/dom7kim/Chat_with_Images.git
cd Chat_with_Images
```

3. Create an `env.list` file in the project directory. This file should contain your OpenAI API key and HuggingFace API token:
```makefile
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```
> Make sure to replace `YOUR_OPENAI_API_KEY` and `YOUR_HUGGINGFACE_TOKEN` with your actual keys.
> Note: Do not enclose your keys in quotes.

4. Build the Docker image:
```bash
docker build -t your_image_name .
```

5. Run the Docker container, ensuring that you provide the `env.list` file:
```bash
docker run --env-file env.list -p 8000:8000 your_image_name
```

6. Access the chatbot by opening your web browser and navigating to [http://localhost:8000](http://localhost:8000).

Utilizing Docker allows you to run the chatbot within a controlled environment, eliminating concerns about dependency conflicts with your system's configuration.






