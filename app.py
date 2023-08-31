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

    def __init__(self, model='gpt-3.5-turbo', temperature = 1.0):
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
        return self.df_imageDB[['file_name', 'caption']].to_string()

    def check_image_file_path(self, inp):
        if os.path.exists(inp) and os.path.isfile(inp):
            file_path = inp
        elif (self.df_imageDB['file_name'] == inp).any():
            file_path = self.df_imageDB[self.df_imageDB['file_name'] == inp]['file_path'].iloc[0]
        else:
            return "Image not found."
        return file_path

    def display_image(self, inp):
        file_path = self.check_image_file_path(inp)
        if file_path == "Image not found.":
            return file_path 
        return f"display {file_path}"

    def get_image_caption(self, image_path):
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
        result = ""
        for keyword in inp.split(','):
            search_vector = get_embedding(keyword, engine="text-embedding-ada-002")
            temp_df = self.df_imageDB[['file_name', 'caption']].copy()
            temp_df['relevance_score'] = self.df_imageDB['embedding'].apply(lambda x: cosine_similarity(x, search_vector))
            temp_df = temp_df.sort_values("relevance_score", ascending=False)
            result = result + "Result for {}:\n{}\n\n".format(keyword, temp_df.to_string())

        return result

    def add_text(self, history, text):
        history = history + [(text, '')]
        return history, ""

    def generate_response(self, history):         
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
            history[-1] = ((file_path,), "[ {} ] - {}".format(file_name, caption))
        else:
            query = "Target Image: {} \n Query: {}".format(file_name, history[-1][0])
            response = self.agent_executor(query)
            response_msg = response["output"]
            history[-1][1] = "[ {} ]".format(file_name)
            history = history + [((file_path,), response_msg)]

        return history, history

    def setup_interface(self):
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
        self.setup_interface()
        self.demo.queue()
        self.demo.launch(server_name='0.0.0.0', server_port=8000)

if __name__ == "__main__":
    # Instantiate and launch
    app = ChatbotApp()
    app.launch()
