def save_dat_file(filepath, data):
    with open(filepath, 'w') as f:
        for i, value in enumerate(data):
            f.write(f"{i} {value}\n")