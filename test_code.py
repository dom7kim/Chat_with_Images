import re
from app import ChatbotApp

class ImageClass:
    def __init__(self, path):
        self._path = path

    @property
    def name(self):
        return self._path
    
test_app = ChatbotApp(temperature=0)

# Scenario 1 tests adding an image with no accompanying user input.
def test_add_file_scenario_1():
    image_instance = ImageClass("test.jpeg")
    history = [["", ""]]
    result_add = test_app.add_file(history, image_instance)
    result_search = test_app.search_relevant_image('pizza')
    assert (result_add[1][0][1] == "A new image named 'test.jpeg' has been successfully uploaded.")
    assert (re.split('\s+',result_search.split('\n')[2])[:2]==['0', 'test.jpeg'])

# Scenario 2 tests adding an image with specific user input.
def test_add_file_scenario_2():
    image_instance = ImageClass("test.jpeg")
    history = [["Count the number of sandwiches in this image. Return only the integer number. Nothing else.", ""]]
    result_add = test_app.add_file(history, image_instance)
    result_search = test_app.search_relevant_image('pizza')
    assert (result_add[1][1][1]=='2')
    assert (re.split('\s+',result_search.split('\n')[2])[:2]==['0', 'test.jpeg'])

if __name__ == "__main__":
    test_add_file_scenario_1()
    test_add_file_scenario_2()
