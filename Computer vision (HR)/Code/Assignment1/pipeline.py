import torch

from segmentation.connected_components import *
from segmentation.main import segment_characters_complete
from util.hebrew_character_gen import HebrewCharacterGen
from segmentation.invert_image import invert_image_8bit

from classification.model import ConvNet

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH


class HebrewPipeline:

    def __init__(self,
                 model_path: str,
                 name_unicode_path: str,
                 label_char_names: str,
                 ) -> None:
        self.model = ConvNet(27)
        self.load_model(model_path)
        self.char_generator = HebrewCharacterGen(name_unicode_path)
        self.char_generator.load_int_name_dict(label_char_names)


    def load_model(self, path: str) -> None:
        """

        :param path:
        :return:
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print("model loaded!")

    def segment_scroll(self, path: str) -> np.ndarray:
        """

        :param path:
        :return:
        """
        return segment_characters_complete(path)
    @staticmethod
    def reshape_one_image(image: np.ndarray, size=(28, 28)) -> np.ndarray:
        desired_width = size[0]
        desired_height = size[1]
        original_height, original_width = image.shape[:2]

        # Calculate new dimensions while maintaining aspect ratio
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            new_width = desired_width
            new_height = int(original_height * desired_width / original_width)
        else:
            new_height = desired_height
            new_width = int(original_width * desired_height / original_height)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))

        # Create a white canvas of the desired size
        canvas = 255 * np.ones((desired_height, desired_width))#, dtype=np.uint8)

        # Calculate the position to paste the resized image
        y_offset = (desired_height - new_height) // 2
        x_offset = (desired_width - new_width) // 2
        # Paste the resized image onto the canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        inverted_canvas = invert_image_8bit(canvas)

        return canvas



    @staticmethod
    def reshape_images(image_array: np.ndarray, size=(28, 28)) -> np.ndarray:
        """

        :param character_array:
        :param size:
        :return:
        """
        result = []
        for line in image_array:
            temp = []
            for image in line:
                temp.append(HebrewPipeline.reshape_one_image(image))
            result.append(temp)
        return result

    @staticmethod
    def reverse_image(image: np.ndarray) -> np.ndarray:
        """

        :param image:
        :return:
        """
        pass

    def classifiy_chars(self, character_array: np.ndarray) -> np.ndarray:
        """

        :param character_array:
        :return:
        """
        res = []
        for line in character_array:
            input = torch.tensor(np.array(line) / 255, dtype=torch.float32)
            input = torch.reshape(input, (len(line),1, 28, 28))
            predictions = self.model.forward(input)
            predicted = predictions.argmax(dim=1, keepdim=True)
            res.append(predicted)
        return res

    def generate_string(self, character_array: np.ndarray) -> str:
        """
        :param character_array:
        :return:
        """
        res = ""
        for line in character_array:
            for character in reversed(line):
                res += self.char_generator.give_char(int(*character))
            res += "\n"
        return res

    def process_scroll(self, input_path: str, output_path: str, gen_docx: bool) -> None:
        segmented_chars = self.segment_scroll(input_path)
        reshaped_chars = HebrewPipeline.reshape_images(segmented_chars)
        # print(segmented_chars.shape)
        labels = self.classifiy_chars(reshaped_chars)
        char_string = self.generate_string(labels)
        # print(labels)
        if char_string is not None:
            # print(char_string)
            with open(f"{output_path}_characters.txt", "w") as file:
                file.write(char_string)
            if gen_docx:
                document = Document()
                paragraph = document.add_paragraph(char_string)
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT

                document.save(f"{output_path}.docx")

    def gridsearch_process_scroll(self, image_dictionary: dict[str, np.ndarray]):
        for settings, images in image_dictionary.items():
            reshaped_chars = HebrewPipeline.reshape_images(images)
            labels = self.classifiy_chars(reshaped_chars)
            char_string = self.generate_string(labels)
            if char_string is not None:
                with open(f"grid_search/{settings}_characters.txt", "w") as file:
                    file.write(char_string)


if __name__ == "__main__":
    pipeline = HebrewPipeline(model_path="conv-0", name_unicode_path="name_unicode.json",
                              label_char_names="../classification/labels_char_names.json")
    path = "data/test_data/124-Fg004.jpg"

    pipeline.process_scroll(path)