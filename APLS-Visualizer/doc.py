import os

def list_files_without_extension(dir, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(dir):
            for name in files:
                file_without_ext = os.path.splitext(name)[0]
                f.write(file_without_ext + '\n')


list_files_without_extension('/Users/ayaan/SPIN_RoadMapper/data/spacenet/train_crops/images', '/Users/ayaan/SPIN_RoadMapper/data/spacenet/train_crops.txt')
list_files_without_extension('/Users/ayaan/SPIN_RoadMapper/data/spacenet/val_crops/images', '/Users/ayaan/SPIN_RoadMapper/data/spacenet/val_crops.txt')