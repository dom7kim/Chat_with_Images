import os  
import re
import pandas as pd

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from openai.embeddings_utils import get_embedding, cosine_similarity

from langchain.agents import Tool, AgentExecutor, ConversationalChatAgent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import pdb
import gradio as gr

class ChatbotApp:
    """
    A chatbot application class that interfaces with Gradio and supports image uploads, 
    object detection, image captioning, and image searching.
    """

    def __init__(self, model='gpt-3.5-turbo', temperature = 1.0):
        """
        Initializes the ChatbotApp instance.
        
        Parameters:
        - model (str): Name of the chat model to use.
        - temperature (float): Sampling temperature for the chat model.
        """
        self.df_imageDB = pd.DataFrame(columns=["file_name", "caption", "embedding", "file_path"])
        
        # Initialize LLM and memory
        self.llm = ChatOpenAI(temperature=temperature, streaming=True, model=model)
        self.memory = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history", output_key="output")
        
        # Define tools
        self.tools = self.define_tools()
        
        # Set up chat_agent and agent_executor
        self.chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=self.llm, tools=self.tools)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.chat_agent,
            tools=self.tools,
            memory=self.memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            verbose=True
        )

    def define_tools(self):
        """
        Defines and returns a list of tools that the chatbot can use.
        
        Returns:
        - list: A list of Tool objects.
        """
        return [
            Tool(
                name = "Retrieve image database",
                func=self.retrieve_image_DB,
                description="Useful for when you need to determine the path, name and caption of the images in the database." 
            ),
            Tool(
                name = "Display Image",
                func=self.display_image,
                description="Useful for when you need to display the image." \
                    "This will display the image on the chat interface."
            ),
            Tool(
                name = "Detect Objects",
                func=self.detect_objects,
                description="Useful for when you need to detect objects in the image." \
                    "This will return a list of all detected objects in the following format:" \
                    "[x1, y1, x2, y2] class_name confidence_score."
            ),
            Tool(
                name = "Search Relevant Image",
                func=self.search_relevant_image,
                description="Use this tool only when you are asked to search for relevant image according to the input query."\
                    "It will return the relevance scores of the images in our database"\
                    "An input query should be a word or phrase, e.g., food, playing soccer, sunny weather, riding a bike."
            )
        ]

    def retrieve_image_DB(self, _):
        """
        Retrieves the image database as a string.
        
        Returns:
        - str: A string representation of the image database.
        """
        return self.df_imageDB[['file_name', 'caption']].to_string()

    def check_image_file_path(self, inp):
        """
        Checks if the input corresponds to a valid image file path or an image name in the database.
        
        Parameters:
        - inp (str): Either the name of an image file or the path to the file.
        
        Returns:
        - str: The file path if found, otherwise returns "Image not found."
        """
        if os.path.exists(inp) and os.path.isfile(inp):
            file_path = inp
        elif (self.df_imageDB['file_name'] == inp).any():
            file_path = self.df_imageDB[self.df_imageDB['file_name'] == inp]['file_path'].iloc[0]
        else:
            return "Image not found."
        return file_path

    def display_image(self, inp):
        """
        Returns a string to display an image if found in the database or on the file path.
        
        Parameters:
        - inp (str): Either the name of an image file or the path to the file.
        
        Returns:
        - str: A string to display the image if found, otherwise returns "Image not found."
        """
        file_path = self.check_image_file_path(inp)
        if file_path == "Image not found.":
            return file_path 
        return f"display {file_path}"

    def get_image_caption(self, image_path):
        """
        Generates a caption for a given image using the Blip image captioning model.
        
        Parameters:
        - image_path (str): Path to the image.
        
        Returns:
        - str: Generated caption for the image.
        """

        image = Image.open(image_path).convert('RGB')
        model_name = "Salesforce/blip-image-captioning-large"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=100)
        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def detect_objects(self, inp):
        """
        Detects objects in a given image and returns their bounding boxes, class names, and confidence scores.
        
        Parameters:
        - inp (str): Either the name of an image file or the path to the file.
        
        Returns:
        - str: A string containing detected objects' bounding boxes, class names, and confidence scores.
        """
        file_path = self.check_image_file_path(inp)
    
        if file_path == "Image not found.":
            return file_path

        image = Image.open(file_path).convert('RGB')
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))

        return detections

    def search_relevant_image(self, inp):
        """
        Searches for images in the database that are relevant to a given input query.
        
        Parameters:
        - inp (str): Input query which can be a word or phrase.
        
        Returns:
        - str: A string representation of the search results.
        """
        result = ""
        for keyword in inp.split(','):
            search_vector = get_embedding(keyword, engine="text-embedding-ada-002")
            temp_df = self.df_imageDB[['file_name', 'caption']].copy()
            temp_df['relevance_score'] = self.df_imageDB['embedding'].apply(lambda x: cosine_similarity(x, search_vector))
            temp_df = temp_df.sort_values("relevance_score", ascending=False)
            result = result + "Result for {}:\n{}\n\n".format(keyword, temp_df.to_string())

        return result

    def add_text(self, history, text):
        """
        Appends a user's text to the chat history.
        
        Parameters:
        - history (list): Current chat history.
        - text (str): User's input text.
        
        Returns:
        - list: Updated chat history.
        """
        history = history + [(text, '')]
        return history, ""

    def generate_response(self, history):
        """
        Generates a response for the last user's input in the chat history.
        
        Parameters:
        - history (list): Current chat history.
        
        Returns:
        - list: Updated chat history with the generated response.
        """         
        response = self.agent_executor(history[-1][0])
        response_msg = response["output"]
        history[-1][1] = response_msg

        if response['intermediate_steps']:
            for agentaction in response['intermediate_steps']:
                if agentaction[-1].startswith('display'):
                    file_path = agentaction[-1].split(' ')[-1]
                    history = history + [((file_path,), None)]

        return history, history

    def get_unique_file_name(self, file_name):
        """
        Generates a unique file name by appending or incrementing a number if necessary.
        
        Parameters:
        - file_name (str): Original file name.
        
        Returns:
        - str: A unique file name.
        """
        base_name, ext = os.path.splitext(file_name)
        match = re.search(r'(.*)\((\d+)\)$', base_name)
        
        if match:
            base_name_without_number = match.group(1)
            counter = int(match.group(2))
        else:
            base_name_without_number = base_name
            counter = 1

        new_file_name = file_name
        while new_file_name in self.df_imageDB['file_name'].values:
            new_file_name = f"{base_name_without_number}({counter}){ext}"
            counter += 1
    
        return new_file_name

    def add_file(self, history, file):
        """
        Processes an uploaded file, generates a caption, and updates the chat history.
        
        Parameters:
        - history (list): Current chat history.
        - file (obj): Uploaded file object.
        
        Returns:
        - list: Updated chat history with information about the uploaded file.
        """
        file_name = self.get_unique_file_name(file.name.split('/')[-1])
        file_path = file.name  # this gives the path to the file, not just the file's name

        caption = self.get_image_caption(file.name)
        embedding = get_embedding(caption, engine='text-embedding-ada-002')

        # Add data to df_imageDB
        data = {
            "file_name": file_name,
            "caption": "This image depicts: {}".format(caption),
            "embedding": embedding,
            "file_path": file_path
        }
        self.df_imageDB = pd.concat([self.df_imageDB, pd.DataFrame([data])], ignore_index=True)

        if history[-1] == ['', '']: 
            history[-1] = ((file_path,), "A new image named '{}' has been successfully uploaded.".format(file_name))
        else:
            query = "Target Image: {} \n Query: {}".format(file_name, history[-1][0])
            response = self.agent_executor(query)
            response_msg = response["output"]
            history[-1][1] = "[ {} ]".format(file_name)
            history = history + [((file_path,), response_msg)]

        return history, history

    def setup_interface(self):
        """
        Sets up the Gradio interface for the chatbot application.
        """
        self.demo = gr.Blocks()
        
        with self.demo:
            chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=600)
            with gr.Row():
                with gr.Column(scale=0.85):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                        container=False)
                with gr.Column(scale=0.15, min_width=0):
                    btn = gr.UploadButton("Upload 🖼️", file_types=["image"]) 

            state = gr.State()

            txt.submit(self.add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
                self.generate_response, inputs=[chatbot], outputs=[chatbot, state], queue=True)
            btn.upload(self.add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
                self.add_file, [chatbot, btn], outputs=[chatbot, state], queue=False)

    def launch(self):
        """
        Sets up the Gradio interface and then launches the application.
        """
        self.setup_interface()
        self.demo.queue()
        self.demo.launch(server_name='0.0.0.0', server_port=8000)

if __name__ == "__main__":
    # Instantiate and launch
    app = ChatbotApp(model='gpt-4')
    app.launch()
