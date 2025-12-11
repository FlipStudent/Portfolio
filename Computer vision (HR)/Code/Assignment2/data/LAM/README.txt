  _        _    __  __ 
 | |      / \  |  \/  |
 | |     / _ \ | |\/| |
 | |___ / ___ \| |  | |
 |_____/_/   \_\_|  |_|
                       

### Folder structure

LAM
├───lines
│   ├───split -> contains the different splits cited in the paper
│   │   ├───basic
│   │   ├───decades_vs_decade
│   │   └───leave_decade_out
│   │       ├───leave_decade_4_out
│   │       ├───leave_decade_1_out
│   │       ├───leave_decade_6_out
│   │       ├───leave_decade_3_out
│   │       ├───leave_decade_5_out
│   │       └───leave_decade_2_out
│   ├───img -> all image lines cropped
│   └───transcriptions.json -> contains all the samples informations
|                              in one single json file
├───full_pages -> contains the whole pages scans with
|   |             the relative xml files with the bounding
|   |             boxes and label of each line
|   |   
│   ├───002
│   │   ├───xml
│   │   └───img
│   ├─── ...
│   │   
│   └───094
│       ├───xml
│       └───img
└───utils -> contains the script "dummy_main.py" that
    |        implement the LAM Dataset class in pytorch
    |        
    └───dummy_main.py