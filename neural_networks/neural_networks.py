from PIL import Image

# iterate through training list and create dataset of pixels

with open('downgesture_train.list.txt', 'r') as train_data:

	lines = train_data.read().splitlines()
	dataset = []
	for line in lines:
		img = Image.open(line)
		pix_val = list(img.getdata())
		# append label to pixel dataset
		if 'down' in line:
			pix_val.append(1)
		else:
			pix_val.append(0)
		dataset.append(pix_val)

	print(dataset)