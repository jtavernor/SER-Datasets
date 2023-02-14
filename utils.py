def load_json(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return None
    with open(filename) as json_file:
        return json.load(json_file)