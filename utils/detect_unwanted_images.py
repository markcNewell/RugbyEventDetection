import json, os


def import_json(file):
	if os.path.exists(file):
		with open(file) as f:
			data = json.load(f)
			return data
	else:
		raise FileNotFoundException("No such json file")



def get_filenames(indir):
	return os.listdir(indir)




if __name__ == '__main__':
	data = import_json('./results.json')
	keys = list(data.keys())

	files = get_filenames('../utils/done/')

	print(len(keys),len(files))

	for key in keys:
		if key not in files:
			print("json:", key)



	for file in files:
		if file not in keys:
			print("file:", file)
			os.remove('../utils/done/'+file)



