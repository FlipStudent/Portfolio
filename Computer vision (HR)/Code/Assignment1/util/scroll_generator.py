from font_generator import *
from font_transformer import *
import csv
import glob


class FileGenerator():
    def __init__(self, 
                 file_shape, 
                 deformation_strength: int, 
                 fontGenerator: FontGenerator, 
                 fontTransformer: FontTransformer,
                 label_output_loc: str,
                 image_output_loc: str
                 ):
        

        self.file_shape = file_shape
        self.file = np.zeros(shape=self.file_shape)

        self.deform_strength = deformation_strength

        self.fontGenerator = fontGenerator
        self.fontTransformer = fontTransformer

        self.current_line = 1
        self.letter_spacer = -20
        self.start_x = file_shape[1] - self.fontGenerator.out_size

        self.label_loc  = label_output_loc
        self.image_loc = image_output_loc

    def generate_scroll(self, output_name:str="scroll", y_space:int=-10, x_space:int=-25) -> np.array:

        self.current_line = 1
        self.file = np.zeros(shape=self.file_shape)

        self.label_file = os.path.join(self.label_loc, output_name) + '.csv'
        self.image_file = os.path.join(self.image_loc, output_name)

        # Initialize the label file
        col_headers = ['name', 'xmin', 'ymin', 'xmax', 'ymax']
        with open(self.label_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(col_headers)

        # Check if there is room for another line (vertical space)
        while (self.current_line + 1) * self.fontGenerator.out_size < self.file.shape[0]:
            self.generate_line(y_space, x_space)
            self.current_line += 1

        # Overlapping characters have values > 1, here we clip that back to [0,1]
        self.file = np.clip(self.file, a_max=1, a_min=0)

        # See the output image
        # plt.imshow(self.file, cmap="gray")
        # plt.show()

        # Save the output image
        im = Image.fromarray(self.file * 255).convert('RGB')
        im.save(self.image_file + ".png")

        return self.file
    
    def generate_line(self, y_space:int=0, x_space:int=0):
        # No more room for this line
        if ((self.current_line + 1) * self.fontGenerator.out_size) + y_space >= self.file.shape[0]:
            print("Error: page is full")
            return False
        
        # Start typing at after the margin
        x = self.start_x

        # While there is room on the current line:
        while x - (self.fontGenerator.out_size * 2) > 0:
            # 7/8 times write a char, 1/8 times write a space
            if np.random.rand() > 0.15:
                x = self.place_letter(x, x_space, y_space)
            else:
                x = self.place_space(x, x_space)
        return True
    
    def place_letter(self, x:int, x_space:int=0, y_space:int=0):
        
        char = random.sample(list(self.fontGenerator.char_map), 1)[0]
        char_shape = (self.fontGenerator.out_size , self.fontGenerator.out_size)

        char_img = self.fontGenerator.create_image(char, char_shape)
        char_img = self.fontTransformer.elastic_deformation(char_img, 4)
        char_img = self.fontTransformer.cut_line(char_img)

        char_width = char_img.shape[1]
        char_height = char_img.shape[0]

        # space = int( (char_width + 20) / 2)
        space = int( (np.random.normal()*5) + x_space)  

        start_y = int( (self.current_line * char_height) + y_space )
        end_y   = int( start_y + char_height )

        start_x = int( (x-char_width)-space )
        end_x   = int( x-space )

        self.file[start_y:end_y , start_x:end_x] += char_img

        # self.file[x-char_width-space:x-space]
        label = [char, start_x, start_y, end_x, end_y]
        self.add_label(label)

        return x - space - char_width
    
    def place_space(self, x, x_space):
        return x - (np.random.rand() * self.fontGenerator.out_size) + x_space
    
    def add_label(self, char_and_bbox):
        with open(self.label_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(char_and_bbox)
        # Check that it works:

        # start = (char_and_bbox[1], char_and_bbox[2])
        # end = (char_and_bbox[3], char_and_bbox[4])

        # self.file = cv2.rectangle(self.file, start, end, (255, 255, 255), 1)
        # plt.imshow(file)
        # plt.show()
        pass
    
if __name__ == "__main__":
    font_path = './data/font/habbakuk/Habbakuk.TTF'
    letter_size = 50
    fontGenerator = FontGenerator(font_path, letter_size=letter_size, letter_padding=10)
    fontTransformer = FontTransformer()

    file_width = 1000
    file_height = 600

    label_output_loc = './generated_data/labels'
    image_output_loc = './generated_data/images'

    gen = FileGenerator(file_shape = (file_height, file_width),
                        deformation_strength = 4,
                        fontGenerator=fontGenerator,
                        fontTransformer=fontTransformer,
                        label_output_loc=label_output_loc,
                        image_output_loc=image_output_loc
                        )
    
    create_n = 5

    for i in range(create_n):
        files_in_loc = glob.glob(label_output_loc + '/*.csv')
        if not files_in_loc:
            next_number_file = 0
        else:
            latest_file = max(files_in_loc, key=os.path.getctime)
            next_number_file = int(latest_file.split('_')[-1].split('.')[0]) + 1 # equals '2' (str) if the last file was scroll_1.csv
        output_name = "scroll_" + str(next_number_file)
        
        np.random.seed(next_number_file)
        print(f"Generating {output_name}")
        gen.generate_scroll(output_name, y_space=-10, x_space=-25)
