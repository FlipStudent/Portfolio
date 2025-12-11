import os
import matplotlib.pyplot as plt

def count_items_in_folders(root_dir, excluded_folder):
    folder_counts = {}
    for root, dirs, files in os.walk(root_dir):
        if excluded_folder in dirs:
            dirs.remove(excluded_folder)  # Exclude the specified folder
        folder_name = os.path.basename(root)
        file_count = len(files)
        folder_counts[folder_name] = file_count
    return folder_counts

def plot_histogram(folder_counts):
    folders = list(folder_counts.keys())
    counts = list(folder_counts.values())
    plt.figure(figsize=(6,3))
    plt.bar(range(len(folders)), counts, align='center')
    plt.xticks(range(len(folders)), folders, rotation=45, ha='right')
    plt.xlabel('Letter name')
    plt.ylabel('Number of Items')
    # plt.title('Item Count in Each Folder')
    plt.savefig('class_inbalance.png',bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    root_directory = "../data/monkbrill"
    excluded_folder = "monkbrill_reshaped"
    folder_counts = count_items_in_folders(root_directory, excluded_folder)
    print(sum(folder_counts.values()))
    folder_counts.pop('monkbrill')
    plot_histogram(folder_counts)
