import os
import csv
from pathlib import Path

images = []
masks = []

root = Path(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
print(root)

data_folder_path = Path(root, 'data')
print(data_folder_path)
# # test, train or val
# for data_subset in os.listdir(Path(root / data_folder_path)):

#     # image or mask
#     for image_or_mask in os.listdir(Path(root / data_folder_path / data_subset)):

#         # filenames
#         for file in os.listdir(Path(root / data_folder_path / data_subset / image_or_mask)):

#             if image_or_mask == 'image':
#                 images.append(str(data_folder_path / data_subset / image_or_mask / file))
#             elif image_or_mask == 'mask':
#                 masks.append(str(data_folder_path / data_subset / image_or_mask / file))
#             else:
#                 print('Error: folders in data / data subset / are not image or mask.')

# # Zip lists into paired tuples
# data = list(zip(images, masks))

# # Write CSV file
# with open("../data.csv", "w", newline='') as fp:
#     writer = csv.writer(fp, delimiter=",")
#     writer.writerows(data)
